from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Qwen model with TRL RLOO on OpenCodeInstruct")
    parser.add_argument("--model-id", required=True, help="Model path or HF id")
    parser.add_argument(
        "--dataset-path",
        default="opencodeinstruct_0_30000.jsonl",
        help="JSONL exported by get_opencodeinstruct_dataset.py",
    )
    parser.add_argument(
        "--eval-path",
        default="opencodeinstruct_eval_100.jsonl",
        help="JSONL hold-out set used for periodic evaluation diagnostics",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    parser.add_argument("--max-samples", type=int, default=30_000, help="Limit number of training samples")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
    parser.add_argument("--report-to", default=None, help="Comma-separated integrations, e.g. wandb")
    parser.add_argument("--reward-timeout-sec", type=int, default=30)
    parser.add_argument("--train-diagnostics-samples", type=int, default=100)
    parser.add_argument("--eval-diagnostics-samples", type=int, default=100)
    parser.add_argument("--diagnostics-every-steps", type=int, default=100)
    parser.add_argument("--diagnostics-file", default=None, help="JSONL diagnostics output path")
    parser.add_argument("--diagnostics-max-new-tokens", type=int, default=256)
    parser.add_argument("--diagnostics-temperature", type=float, default=0.2)
    parser.add_argument("--diagnostics-top-p", type=float, default=0.95)
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


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


def maybe_build_peft_config(args: argparse.Namespace):
    if not args.use_lora:
        return None
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )


def main() -> None:
    args = parse_args()

    report_to = None
    if args.report_to:
        report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]

    eval_dataset = build_dataset(args.eval_path, split_tag="eval")
    eval_row_ids = {row["row_id"] for row in eval_dataset}
    train_dataset = build_dataset(
        args.dataset_path,
        max_samples=args.max_samples,
        exclude_row_ids=eval_row_ids,
        split_tag="train",
    )
    print(f"Loaded train samples: {len(train_dataset)}")
    print(f"Loaded eval samples: {len(eval_dataset)}")

    diagnostics_path = args.diagnostics_file or str(Path(args.output_dir) / "diagnostics.jsonl")
    train_diagnostics_samples = sample_rows_for_diagnostics(train_dataset, args.train_diagnostics_samples)
    eval_diagnostics_samples = sample_rows_for_diagnostics(eval_dataset, args.eval_diagnostics_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    rloo_args = RLOOConfig(
        output_dir=args.output_dir,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to=report_to,
    )

    reward_fn = make_reward_fn(timeout_sec=args.reward_timeout_sec)
    diagnostics_callback = PeriodicDiagnosticsCallback(
        tokenizer=tokenizer,
        train_samples=train_diagnostics_samples,
        eval_samples=eval_diagnostics_samples,
        reward_timeout_sec=args.reward_timeout_sec,
        every_n_steps=args.diagnostics_every_steps,
        output_path=diagnostics_path,
        max_new_tokens=args.diagnostics_max_new_tokens,
        temperature=args.diagnostics_temperature,
        top_p=args.diagnostics_top_p,
    )
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=rloo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[diagnostics_callback],
        peft_config=maybe_build_peft_config(args),
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
