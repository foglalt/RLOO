# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re
import json
from evaluate import load
import pickle
import sys
from typing import Iterable, Iterator

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

# %%
model_id = '/ssd/bszalontai_local/models_hf/Qwen2.5-Coder-1.5B-Instruct/'
assert os.path.exists(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=False,trust_remote_code=True)
model_name = model_id.strip('/').split('/')[-1]
print(f'Evaluated model: {model_name}')

# %%
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16,
    device_map="auto", 
    trust_remote_code=True,
)


# %%
def qwen_coder_chat(
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
):
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Qwen models use a chat template for proper formatting
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )

    # Drop the prompt tokens and decode only the new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# %%
dataset_path = "opencodeinstruct_0_10000.jsonl"
assert os.path.exists(dataset_path), f"{dataset_path} not found"

df = pd.read_json(dataset_path, lines=True)

descriptions, unit_tests, entry_points = list(df["input"]), list(df["unit_tests"]), list(df["entry_point"])

example_id = 20
description, tests, entry_point = descriptions[example_id], unit_tests[example_id], entry_points[example_id]
print(f'{description}\n{tests}\n{entry_point}')

# %%
prompt_start = 'You are an expert Python coding assistant.\nFollow these rules when solving the task below:\n- Implement the requested function exactly once using the provided signature.\n- Return efficient, idiomatic Python 3 code.\n- Do not include markdown, explanations, tests, or extra helper text—only executable code.\n'
prompt_end = description

response = qwen_coder_chat(prompt_start+prompt_end)
print(response)


# %%
def unwrap_code(text: str) -> str:
    """
    Remove optional <think> blocks, then return the last ```python ... ``` block.
    If none found, return the stripped raw text.
    """
    text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    code_blocks = re.findall(
        r"```python\s*(.*?)\s*```", text_without_think, flags=re.DOTALL
    )
    if code_blocks:
        return code_blocks[-1].strip()
    return text_without_think.strip()

code = unwrap_code(response)
print(code)


UNSAFE_CODE_PATTERNS = [
    ("import_os", re.compile(r"\bimport\s+os\b", re.IGNORECASE)),
    ("os_usage", re.compile(r"\bos\s*\.", re.IGNORECASE)),
    ("subprocess", re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("shutil", re.compile(r"\bshutil\b", re.IGNORECASE)),
    ("pathlib", re.compile(r"\bpathlib\b", re.IGNORECASE)),
    ("file_io", re.compile(r"\b(open|os\.path|pathlib\.Path)\b", re.IGNORECASE)),
]


def detect_environment_access(code_str: str) -> list[str]:
    matches: list[str] = []
    for label, pattern in UNSAFE_CODE_PATTERNS:
        if pattern.search(code_str):
            matches.append(label)
    return matches


# %%
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

ratio, passed, total, errors = evaluate_sample(code,tests,entry_point)
print(f"{ratio} {passed} {total} {errors}")

# %%
results: list[dict[str, object]] = []
processed = 0

begin=0
end=10000

for idx, (instruction, tests, entry_point) in enumerate(
    tqdm(
        list(zip(descriptions[begin:end], unit_tests[begin:end], entry_points[begin:end], strict=True)),
        total=end-begin,
        desc="Opencodeinstruct eval with Qwen2.5-Coder-1.5B",
    )
):  
    if not instruction or not tests:
        continue

    prompt = prompt_start + instruction
    raw_response = qwen_coder_chat(prompt)
    generated_code = unwrap_code(raw_response)
    unsafe_reasons = detect_environment_access(generated_code)
    if unsafe_reasons:
        record = {
            "id": idx,
            "instruction": instruction,
            "generated_code": generated_code,
            "tests_passed": 0,
            "tests_total": len(tests),
            "test_ratio": 0.0,
            "errors": [f"skipped_unsafe: {', '.join(unsafe_reasons)}"],
            "skipped": True,
        }
        results.append(record)
        processed += 1
        continue
    ratio, passed, total, error_messages = evaluate_sample(generated_code, tests, entry_point)
    error_messages = [ f"#{idx} {error}" for error in error_messages]

    record = {
        "id": idx,
        "instruction": instruction,
        "generated_code": generated_code,
        "tests_passed": passed,
        "tests_total": total,
        "test_ratio": ratio,
        "skipped": False,
    }
    if error_messages:
        record["errors"] = error_messages
    results.append(record)
    processed += 1


with open("opencodeinstruct_qwen_eval.jsonl", "w", encoding="utf-8") as fh:
    for rec in results:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

evaluated_records = [rec for rec in results if not rec.get("skipped")]
if evaluated_records:
    mean_ratio = sum(rec["test_ratio"] for rec in evaluated_records) / len(evaluated_records)
else:
    mean_ratio = 0.0
skipped_count = len(results) - len(evaluated_records)
print(
    f"Processed {processed} samples; evaluated {len(evaluated_records)} (skipped {skipped_count})"
    f"; mean unit-test ratio = {mean_ratio:.4f}"
)

# %%
