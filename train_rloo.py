from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    parser.add_argument("--dataset-path", required=True, help="JSONL exported by get_opencodeinstruct_dataset.py")
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of training samples")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
    parser.add_argument("--report-to", default=None, help="Comma-separated integrations, e.g. wandb")
    parser.add_argument("--reward-timeout-sec", type=int, default=30)
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


def build_dataset(dataset_path: str, max_samples: int | None = None) -> Dataset:
    rows: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            instruction = row.get("input")
            tests = row.get("unit_tests")
            entry_point = row.get("entry_point")
            if not instruction or not isinstance(tests, list) or not tests or not entry_point:
                continue
            rows.append(
                {
                    "prompt": PROMPT_PREFIX + instruction,
                    "unit_tests": tests,
                    "entry_point": entry_point,
                    "row_id": row.get("id"),
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

    train_dataset = build_dataset(args.dataset_path, max_samples=args.max_samples)
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
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
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
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=rloo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=maybe_build_peft_config(args),
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
