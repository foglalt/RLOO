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
import ast
import subprocess
import sys

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
dataset_path = "opencodeinstruct_train_29541.jsonl"
assert os.path.exists(dataset_path), f"{dataset_path} not found"

df = pd.read_json(dataset_path, lines=True)

descriptions, unit_tests, entry_points = list(df["input"]), list(df["unit_tests"]), list(df["entry_point"])
if "id" in df.columns:
    row_ids = list(df["id"])
else:
    row_ids = list(range(len(df)))

example_id = 20
description, tests, entry_point = descriptions[example_id], unit_tests[example_id], entry_points[example_id]
print(f'{description}\n{tests}\n{entry_point}')

# %%
prompt_start = 'You are an expert Python coding assistant.\nFollow these rules when solving the task below:\n- Implement the requested function exactly once using the provided signature.\n- Return efficient, idiomatic Python 3 code.\n- Do not include markdown, explanations, tests, or extra helper text—only executable code.\n'
prompt_end = description

response = qwen_coder_chat(prompt_start+prompt_end)
print(response)


# %% Helper filters and execution guards
UNSAFE_CODE_PATTERNS = [
    ("import_os", re.compile(r"\bimport\s+os\b", re.IGNORECASE)),
    ("os_usage", re.compile("\\bos\\s*\\.", re.IGNORECASE)),
    ("subprocess", re.compile(r"\bsubprocess\b", re.IGNORECASE)),
    ("shutil", re.compile(r"\bshutil\b", re.IGNORECASE)),
    ("pathlib", re.compile(r"\bpathlib\b", re.IGNORECASE)),
    ("file_io", re.compile(r"\b(open|os\.path|pathlib\.Path)\b", re.IGNORECASE)),
]

TEST_SKIP_PATTERNS = [
    ("input_required", re.compile(r"\binput\s*\(", re.IGNORECASE)),
    ("file_io", re.compile(r"\b(open|os\.|pathlib\.)", re.IGNORECASE)),
    ("network", re.compile(r"\b(requests|urllib|httpx|http\.client)\b", re.IGNORECASE)),
    ("database", re.compile(r"\bsqlite3\b", re.IGNORECASE)),
    ("server_runtime", re.compile(r"\bFlask\b|app\.run\(|run_server\(", re.IGNORECASE)),
    ("system_monitoring", re.compile(r"\bpsutil\b", re.IGNORECASE)),
    ("heavy_deps", re.compile(r"\b(matplotlib|wordcloud|pandas|numpy|sklearn|nltk)\b", re.IGNORECASE)),
    ("random_maze", re.compile(r"\b(random|randrange)\b.*\bmaze\b|\bmaze\b.*\b(random|randrange)\b", re.IGNORECASE)),
    ("sleep_or_loop", re.compile(r"\btime\.sleep\b|while\s+True", re.IGNORECASE)),
]
PROMPT_SKIP_PATTERNS = [
    ("file_or_dir_task", re.compile(r"\b(file|files|directory|directories|filesystem|path|folder|os\.walk|walk through)\b", re.IGNORECASE)),
    ("api_or_server_task", re.compile(r"\b(restful|flask|fastapi|endpoint|api server|web server)\b", re.IGNORECASE)),
    ("database_task", re.compile(r"\b(database|sqlite|sql|mongodb|postgres)\b", re.IGNORECASE)),
]


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


def keep_necessary(code: str) -> str:
    """Keep only imports and class/function definitions (with their bodies)."""
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "\n".join([ln for ln in lines if ln.lstrip().startswith(("import ", "from "))])

    keep: set[int] = set()

    def mark_span(node: ast.AST):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None:
            return
        for ln in range(start, (end or start) + 1):
            keep.add(ln)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark_span(node)

    filtered = [line for idx, line in enumerate(lines, start=1) if idx in keep]
    return "\n".join(filtered)


def detect_environment_access(code_str: str) -> list[str]:
    matches: list[str] = []
    for label, pattern in UNSAFE_CODE_PATTERNS:
        if pattern.search(code_str):
            matches.append(label)
    return matches


def detect_test_issues(unit_tests: list[str]) -> list[str]:
    """Detect tests that require external resources or interactive input."""
    combined = "\n".join(unit_tests)
    matches: list[str] = []
    for label, pattern in TEST_SKIP_PATTERNS:
        if pattern.search(combined):
            matches.append(label)
    return matches


def detect_prompt_issues(prompt: str) -> list[str]:
    matches: list[str] = []
    for label, pattern in PROMPT_SKIP_PATTERNS:
        if pattern.search(prompt):
            matches.append(label)
    return matches


ISOLATED_EVAL_SCRIPT = r"""
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


code = unwrap_code(response)
print(code)


# %%
def evaluate_sample(code_str: str, unit_tests: list[str], entry_point: str | None) -> tuple[float, int, int, list[str]]:
    payload = {
        "code_str": code_str,
        "unit_tests": unit_tests,
        "entry_point": entry_point,
    }
    timeout_sec = max(8, 3 * len(unit_tests))

    try:
        completed = subprocess.run(
            [sys.executable, "-c", ISOLATED_EVAL_SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0.0, 0, len(unit_tests), [f"timeout_after_{timeout_sec}s"]
    except Exception as exc:  # pylint: disable=broad-except
        return 0.0, 0, len(unit_tests), [f"evaluator_exec_error: {exc!r}"]

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return 0.0, 0, len(unit_tests), [f"evaluator_process_error: {stderr or 'non-zero exit'}"]

    output = (completed.stdout or "").strip()
    if not output:
        return 0.0, 0, len(unit_tests), ["evaluator_protocol_error: empty_stdout"]

    try:
        result = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError:
        return 0.0, 0, len(unit_tests), [f"evaluator_protocol_error: invalid_json: {output[:200]}"]

    total = int(result.get("total", len(unit_tests)))
    if result.get("status") == "code_error":
        return 0.0, 0, total, [f"code_exec_error: {result.get('error', 'unknown')}"]

    passed = int(result.get("passed", 0))
    errors = result.get("errors", [])
    if not isinstance(errors, list):
        errors = [f"evaluator_protocol_error: invalid_errors_field: {errors!r}"]
    ratio = passed / total if total else 0.0
    return ratio, passed, total, errors

ratio, passed, total, errors = evaluate_sample(code,tests,entry_point)
print(f"{ratio} {passed} {total} {errors}")

# %%
results: list[dict[str, object]] = []
processed = 0

begin=0
end=10000

eval_total = min(end, len(descriptions)) - begin
for row_id, instruction, tests, entry_point in tqdm(
    zip(
        row_ids[begin:end],
        descriptions[begin:end],
        unit_tests[begin:end],
        entry_points[begin:end],
        strict=True,
    ),
    total=eval_total,
    desc="Opencodeinstruct eval with Qwen2.5-Coder-1.5B",
):
  
    if not instruction or not tests:
        continue

    if not entry_point:
        record = {
            "id": row_id,
            "instruction": instruction,
            "generated_code": "",
            "tests_passed": 0,
            "tests_total": len(tests),
            "test_ratio": 0.0,
            "errors": ["skipped_missing_entry_point"],
            "skipped": True,
        }
        results.append(record)
        processed += 1
        continue

    test_skip_reasons = detect_test_issues(tests)
    prompt_skip_reasons = detect_prompt_issues(instruction)
    all_skip_reasons = test_skip_reasons + [f"prompt_{reason}" for reason in prompt_skip_reasons]
    if all_skip_reasons:
        record = {
            "id": row_id,
            "instruction": instruction,
            "generated_code": "",
            "tests_passed": 0,
            "tests_total": len(tests),
            "test_ratio": 0.0,
            "errors": [f"skipped_task_issue: {', '.join(all_skip_reasons)}"],
            "skipped": True,
        }
        results.append(record)
        processed += 1
        continue

    prompt = prompt_start + instruction
    raw_response = qwen_coder_chat(prompt)
    generated_code = keep_necessary(unwrap_code(raw_response))
    unsafe_reasons = detect_environment_access(generated_code)
    if unsafe_reasons:
        record = {
            "id": row_id,
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
    error_messages = [f"#{row_id} {error}" for error in error_messages]

    record = {
        "id": row_id,
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
