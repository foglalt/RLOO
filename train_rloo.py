from __future__ import annotations

import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import RLOOConfig, RLOOTrainer


PROMPT_PREFIX = (
    "You are an expert Python coding assistant.\n"
    "Follow these rules when solving the task below:\n"
    "- Implement the requested function exactly once using the provided signature.\n"
    "- Return efficient, idiomatic Python 3 code.\n"
    "- Do not include markdown, explanations, tests, or extra helper text, only executable code.\n"
)

ISOLATED_REWARD_SCRIPT = r"""
import contextlib
import io
import json
import sys

def main():
    payload = json.loads(sys.stdin.read())
    code_str = payload["code_str"]
    unit_tests = payload["unit_tests"]
    entry_point = payload.get("entry_point")

    ns = {}
    errors = []
    passed = 0
    total = len(unit_tests)

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code_str, ns)
    except Exception as exc:
        print(json.dumps({"status": "code_error", "error": f"{exc!r}", "total": total}))
        return

    if entry_point and entry_point in ns and callable(ns[entry_point]):
        ns["candidate"] = ns[entry_point]

    for idx, test_snippet in enumerate(unit_tests):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(test_snippet, ns)
            passed += 1
        except Exception as exc:
            errors.append(f"test_{idx}_error: {exc!r}")

    print(json.dumps({"status": "ok", "passed": passed, "total": total, "errors": errors}))

if __name__ == "__main__":
    main()
"""


# Training configuration (edit these values directly).
MODEL_ID = "/ssd/bszalontai_local/models_hf/Qwen2.5-Coder-1.5B-Instruct/"
DATASET_PATH = "opencodeinstruct_0_30000.jsonl"
EVAL_PATH = "opencodeinstruct_eval_100.jsonl"
OUTPUT_DIR = "runs/qwen_rloo"
MAX_SAMPLES = 30_000
LEARNING_RATE = 1e-6
MAX_STEPS = 200
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 768
MAX_COMPLETION_LENGTH = 256
TEMPERATURE = 0.7
TOP_P = 0.95
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100
SEED = 42
USE_BF16 = False
REPORT_TO = None  # Example: ["wandb"]
REWARD_TIMEOUT_SEC = 30
TRAIN_DIAGNOSTICS_SAMPLES = 100
EVAL_DIAGNOSTICS_SAMPLES = 100
DIAGNOSTICS_EVERY_STEPS = 100
DIAGNOSTICS_FILE = None  # Defaults to OUTPUT_DIR/diagnostics.jsonl when None.
DIAGNOSTICS_MAX_NEW_TOKENS = 256
DIAGNOSTICS_TEMPERATURE = 0.2
DIAGNOSTICS_TOP_P = 0.95
USE_LORA = False
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def unwrap_code(text: str) -> str:
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    code_blocks = re.findall(r"```python\s*(.*?)\s*```", text_without_think, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return text_without_think.strip()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(str(block.get("text", "")))
                        elif "text" in block:
                            text_parts.append(str(block["text"]))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts)
    return str(completion)


def evaluate_code(code_str: str, unit_tests: list[str], entry_point: str, timeout_sec: int) -> tuple[float, list[str]]:
    payload = {"code_str": code_str, "unit_tests": unit_tests, "entry_point": entry_point}
    total = len(unit_tests)
    if total == 0:
        return 0.0, ["empty_tests"]

    try:
        completed = subprocess.run(
            [sys.executable, "-c", ISOLATED_REWARD_SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0.0, [f"timeout_after_{timeout_sec}s"]
    except Exception as exc:  # pylint: disable=broad-except
        return 0.0, [f"reward_eval_error: {exc!r}"]

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return 0.0, [f"reward_eval_process_error: {stderr or 'non-zero exit'}"]

    output = (completed.stdout or "").strip()
    if not output:
        return 0.0, ["reward_eval_protocol_error: empty_stdout"]

    try:
        result = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError:
        return 0.0, [f"reward_eval_protocol_error: invalid_json: {output[:200]}"]

    if result.get("status") == "code_error":
        return 0.0, [f"code_exec_error: {result.get('error', 'unknown')}"]

    passed = int(result.get("passed", 0))
    out_total = int(result.get("total", total))
    ratio = passed / out_total if out_total else 0.0
    errors = result.get("errors", [])
    if not isinstance(errors, list):
        errors = [f"reward_eval_protocol_error: invalid_errors_field: {errors!r}"]
    return ratio, errors


def build_dataset(
    dataset_path: str,
    max_samples: int | None = None,
    exclude_row_ids: set[Any] | None = None,
    split_tag: str = "train",
) -> Dataset:
    rows: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {dataset_path}: {exc}") from exc

            instruction = row.get("input")
            tests = row.get("unit_tests")
            entry_point = row.get("entry_point")
            row_id = row.get("id")

            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"Invalid row at line {line_no} in {dataset_path}: missing_input (id={row.get('id')!r})"
                )
            if not isinstance(tests, list) or not tests:
                raise ValueError(
                    f"Invalid row at line {line_no} in {dataset_path}: invalid_unit_tests (id={row.get('id')!r})"
                )
            if not all(isinstance(test, str) and test.strip() for test in tests):
                raise ValueError(
                    f"Invalid row at line {line_no} in {dataset_path}: malformed_unit_test_entry (id={row.get('id')!r})"
                )
            if not isinstance(entry_point, str) or not entry_point.strip():
                raise ValueError(
                    f"Invalid row at line {line_no} in {dataset_path}: missing_entry_point (id={row.get('id')!r})"
                )
            if row_id is None:
                raise ValueError(
                    f"Invalid row at line {line_no} in {dataset_path}: missing_id"
                )
            if exclude_row_ids is not None and row_id in exclude_row_ids:
                continue

            rows.append(
                {
                    "prompt": PROMPT_PREFIX + instruction,
                    "unit_tests": tests,
                    "entry_point": entry_point,
                    "row_id": row_id,
                    "split": split_tag,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError("No usable samples found in dataset_path")
    return Dataset.from_list(rows)


def make_reward_fn(timeout_sec: int):
    def reward_fn(prompts, completions, unit_tests, entry_point, **kwargs):  # noqa: ANN001
        del prompts, kwargs
        rewards: list[float] = []
        for completion, tests, target in zip(completions, unit_tests, entry_point, strict=True):
            text = completion_to_text(completion)
            code = unwrap_code(text)
            score, _ = evaluate_code(code, tests, target, timeout_sec=timeout_sec)
            rewards.append(float(score))
        return rewards

    return reward_fn


def sample_rows_for_diagnostics(dataset: Dataset, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    size = min(limit, len(dataset))
    return [dataset[i] for i in range(size)]


def generate_completion(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def compute_diagnostics_for_split(
    model,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    reward_timeout_sec: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[float, float, int]:
    if not samples:
        return 0.0, 0.0, 0

    rewards: list[float] = []
    successes = 0
    for sample in samples:
        raw_completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=sample["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        code = unwrap_code(raw_completion)
        reward, _ = evaluate_code(
            code_str=code,
            unit_tests=sample["unit_tests"],
            entry_point=sample["entry_point"],
            timeout_sec=reward_timeout_sec,
        )
        reward = float(reward)
        rewards.append(reward)
        if reward >= 0.999999:
            successes += 1

    avg_reward = sum(rewards) / len(rewards)
    success_ratio = successes / len(rewards)
    return avg_reward, success_ratio, len(rewards)


class PeriodicDiagnosticsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        train_samples: list[dict[str, Any]],
        eval_samples: list[dict[str, Any]],
        reward_timeout_sec: int,
        every_n_steps: int,
        output_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.train_samples = train_samples
        self.eval_samples = eval_samples
        self.reward_timeout_sec = reward_timeout_sec
        self.every_n_steps = max(1, every_n_steps)
        self.output_path = Path(output_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._last_logged_step = -1
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, state, model) -> None:
        step = int(state.global_step)
        if step == self._last_logged_step:
            return

        was_training = model.training
        model.eval()
        try:
            train_avg_reward, train_success_ratio, train_count = compute_diagnostics_for_split(
                model=model,
                tokenizer=self.tokenizer,
                samples=self.train_samples,
                reward_timeout_sec=self.reward_timeout_sec,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            eval_avg_reward, eval_success_ratio, eval_count = compute_diagnostics_for_split(
                model=model,
                tokenizer=self.tokenizer,
                samples=self.eval_samples,
                reward_timeout_sec=self.reward_timeout_sec,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        finally:
            if was_training:
                model.train()

        record = {
            "step": step,
            "train_avg_reward": train_avg_reward,
            "train_success_ratio": train_success_ratio,
            "train_samples": train_count,
            "eval_avg_reward": eval_avg_reward,
            "eval_success_ratio": eval_success_ratio,
            "eval_samples": eval_count,
        }
        with open(self.output_path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[diagnostics] step={step} "
            f"train_avg_reward={train_avg_reward:.4f} train_success={train_success_ratio:.4f} "
            f"eval_avg_reward={eval_avg_reward:.4f} eval_success={eval_success_ratio:.4f}"
        )
        self._last_logged_step = step

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args
        if not state.is_world_process_zero:
            return control
        if state.global_step <= 0:
            return control
        if state.global_step % self.every_n_steps != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self._record(state, model)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, control
        if not state.is_world_process_zero:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._record(state, model)


def maybe_build_peft_config():
    if not USE_LORA:
        return None
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules="all-linear",
    )


def main() -> None:
    eval_dataset = build_dataset(EVAL_PATH, split_tag="eval")
    eval_row_ids = {row["row_id"] for row in eval_dataset}
    train_dataset = build_dataset(
        DATASET_PATH,
        max_samples=MAX_SAMPLES,
        exclude_row_ids=eval_row_ids,
        split_tag="train",
    )
    print(f"Loaded train samples: {len(train_dataset)}")
    print(f"Loaded eval samples: {len(eval_dataset)}")

    diagnostics_path = DIAGNOSTICS_FILE or str(Path(OUTPUT_DIR) / "diagnostics.jsonl")
    train_diagnostics_samples = sample_rows_for_diagnostics(train_dataset, TRAIN_DIAGNOSTICS_SAMPLES)
    eval_diagnostics_samples = sample_rows_for_diagnostics(eval_dataset, EVAL_DIAGNOSTICS_SAMPLES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if USE_BF16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    rloo_args = RLOOConfig(
        output_dir=OUTPUT_DIR,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        learning_rate=LEARNING_RATE,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        seed=SEED,
        bf16=USE_BF16,
        remove_unused_columns=False,
        report_to=REPORT_TO,
    )

    reward_fn = make_reward_fn(timeout_sec=REWARD_TIMEOUT_SEC)
    diagnostics_callback = PeriodicDiagnosticsCallback(
        tokenizer=tokenizer,
        train_samples=train_diagnostics_samples,
        eval_samples=eval_diagnostics_samples,
        reward_timeout_sec=REWARD_TIMEOUT_SEC,
        every_n_steps=DIAGNOSTICS_EVERY_STEPS,
        output_path=diagnostics_path,
        max_new_tokens=DIAGNOSTICS_MAX_NEW_TOKENS,
        temperature=DIAGNOSTICS_TEMPERATURE,
        top_p=DIAGNOSTICS_TOP_P,
    )
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=rloo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[diagnostics_callback],
        peft_config=maybe_build_peft_config(),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
