"""Evaluate Qwen-style models on the exported OpenCodeInstruct dataset.

The script mirrors the behaviour of ``qwenTest.py`` but works with the JSONL
files produced by ``get_opencodeinstruct_dataset.py``. It can run on any slice
of the dataset and reports the per-item ratio of unit tests passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable, Iterator

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

DEFAULT_MODEL_ID = "/ssd/bszalontai_local/models_hf/Qwen2.5-Coder-1.5B-Instruct/"
PROMPT_PREFIX = (
    "You are an expert Python coding assistant.\n"
    "Follow these rules when solving the task below:\n"
    "- Implement the requested function exactly once using the provided signature.\n"
    "- Return efficient, idiomatic Python 3 code.\n"
    "- Do not include markdown, explanations, tests, or extra helper text—only executable code.\n"
)
FUNC_NAME_RE = re.compile(r"assert\s+([a-zA-Z_]\w*)\s*\(")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen on OpenCodeInstruct JSONL")
    parser.add_argument("dataset_path", help="Path to opencodeinstruct_*.jsonl produced earlier")
    parser.add_argument(
        "--output-jsonl",
        default="opencodeinstruct_qwen_eval.jsonl",
        help="Where to store per-sample evaluation records",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model name or local path (default: %(default)s)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many dataset rows before evaluation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to evaluate (default: entire file)",
    )
    return parser.parse_args(argv)


def unwrap_code(text: str) -> str:
    """Extract the final ```python``` block, mirroring qwenTest.py behaviour."""

    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    code_blocks = re.findall(r"```python\s*(.*?)\s*```", text_without_think, flags=re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()
    return text_without_think.strip()


def normalize_unit_tests(raw_tests: str | list[str] | None) -> list[str]:
    if raw_tests is None:
        return []
    if isinstance(raw_tests, list):
        return [t for t in raw_tests if isinstance(t, str) and t.strip()]
    if isinstance(raw_tests, str):
        try:
            parsed = json.loads(raw_tests)
            if isinstance(parsed, list):
                return [t for t in parsed if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass
        return [raw_tests]
    return []


def iter_dataset(path: str, start: int = 0, limit: int | None = None) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx < start:
                continue
            if limit is not None and idx >= start + limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def qwen_coder_chat(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_sample(code_str: str, unit_tests: list[str], entry_point: str | None) -> tuple[float, int, int, list[str]]:
    ns: dict[str, object] = {}
    errors: list[str] = []
    try:
        exec(code_str, ns)
    except Exception as exc:  # pylint: disable=broad-except
        return 0.0, 0, len(unit_tests), [f"code_exec_error: {exc!r}"]

    if entry_point and entry_point in ns and callable(ns[entry_point]):
        ns["candidate"] = ns[entry_point]

    passed = 0
    for idx, test_snippet in enumerate(unit_tests):
        try:
            exec(test_snippet, ns)
            passed += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"test_{idx}_error: {exc!r}")

    total = len(unit_tests)
    ratio = passed / total if total else 0.0
    return ratio, passed, total, errors


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(args.dataset_path)
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive when provided")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, add_eos_token=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    results: list[dict[str, object]] = []
    processed = 0

    stream = iter_dataset(args.dataset_path, start=args.start_index, limit=args.limit)
    for row in tqdm(stream, desc="OpenCodeInstruct eval", unit="sample"):
        instruction = row.get("input") or ""
        entry_point = row.get("entry_point")
        unit_tests = normalize_unit_tests(row.get("unit_tests"))
        if not instruction or not unit_tests:
            continue

        prompt = PROMPT_PREFIX + instruction
        raw_response = qwen_coder_chat(
            tokenizer,
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        generated_code = unwrap_code(raw_response)
        ratio, passed, total, error_messages = evaluate_sample(generated_code, unit_tests, entry_point)

        record = {
            "id": row.get("id"),
            "instruction": instruction,
            "generated_code": generated_code,
            "tests_passed": passed,
            "tests_total": total,
            "test_ratio": ratio,
        }
        if error_messages:
            record["errors"] = error_messages
        results.append(record)
        processed += 1

    with open(args.output_jsonl, "w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if results:
        mean_ratio = sum(rec["test_ratio"] for rec in results) / len(results)
    else:
        mean_ratio = 0.0
    print(f"Processed {processed} samples; mean unit-test ratio = {mean_ratio:.4f}")


if __name__ == "__main__":
    main()
