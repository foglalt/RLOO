import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re
import json
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

# +
model_id = '/ssd/bszalontai_local/models_hf/Qwen2.5-Coder-1.5B-Instruct/'
assert os.path.exists(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=False,trust_remote_code=True)
model_name = model_id.strip('/').split('/')[-1]
print(f'Evaluated model: {model_name}')
# -

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.bfloat16,
    device_map="auto", 
    trust_remote_code=True,
)


def qwen_coder_chat(
    prompt: str,
    max_new_tokens: int = 512,
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


# +
humaneval_path = "humanevalpack.jsonl"
assert os.path.exists(humaneval_path), f"{humaneval_path} not found"

humaneval_df = pd.read_json(humaneval_path, lines=True)

descriptions, test_codes, entry_points = list(humaneval_df["instruction"]), list(humaneval_df["test"]), list(humaneval_df["entry_point"])
if "task_id" in humaneval_df.columns:
    task_ids = list(humaneval_df["task_id"])
else:
    task_ids = list(range(len(descriptions)))

# +
prompt_start = 'You are an expert Python coding assistant.\nFollow these rules when solving the task below:\n- Implement the requested function exactly once using the provided signature.\n- Return efficient, idiomatic Python 3 code.\n- Do not include markdown, explanations, tests, or extra helper text—only executable code.\n'
# +
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


# +
def evaluate_single_sample(code_str: str, test_code: str, entry_point: str):
    """
    Execute generated code and then the provided test code.
    Returns:
        score: 1.0 if all tests pass, 0.0 otherwise
        error: optional error string (None if all tests pass)
    """
    isolated_script = r"""
import contextlib
import io
import json
import sys

def main():
    payload = json.loads(sys.stdin.read())
    code_str = payload["code_str"]
    test_code = payload["test_code"]
    entry_point = payload.get("entry_point")

    ns = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code_str, ns)
    except Exception as exc:
        print(json.dumps({"status": "code_error", "error": f"{exc!r}"}))
        return

    if entry_point and entry_point in ns and callable(ns[entry_point]):
        ns["candidate"] = ns[entry_point]

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(test_code, ns)
        print(json.dumps({"status": "pass"}))
    except Exception as exc:
        print(json.dumps({"status": "test_error", "error": f"{exc!r}"}))

if __name__ == "__main__":
    main()
"""

    payload = {
        "code_str": code_str,
        "test_code": test_code,
        "entry_point": entry_point,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", isolated_script],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 0.0, "test_exec_error: timeout_after_15s"
    except Exception as exc:  # pylint: disable=broad-except
        return 0.0, f"test_exec_error: evaluator_exec_error: {exc!r}"

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return 0.0, f"test_exec_error: evaluator_process_error: {stderr or 'non-zero exit'}"

    output = (completed.stdout or "").strip()
    if not output:
        return 0.0, "test_exec_error: evaluator_protocol_error: empty_stdout"

    try:
        result = json.loads(output.splitlines()[-1])
    except json.JSONDecodeError:
        return 0.0, f"test_exec_error: evaluator_protocol_error: invalid_json: {output[:200]}"

    status = result.get("status")
    if status == "pass":
        return 1.0, None
    if status == "code_error":
        return 0.0, f"code_exec_error: {result.get('error', 'unknown')}"
    return 0.0, f"test_exec_error: {result.get('error', 'unknown')}"

# +
results = []
output_jsonl = "humaneval_qwen2_5_instruction_eval.jsonl"

for idx, (instruction, test_code, entry_point, task_id) in enumerate(
    tqdm(
        zip(descriptions, test_codes, entry_points, task_ids, strict=True),
        total=len(descriptions),
        desc="HumanEval instruction eval with Qwen2.5-Coder-1.5B",
    )
):
    prompt = prompt_start + instruction
    raw_response = qwen_coder_chat(prompt)
    generated_code = unwrap_code(raw_response)

    score, error = evaluate_single_sample(generated_code, test_code, entry_point)

    rec = {
        "id": task_id,
        "instruction": instruction,
        "generated_code": generated_code,
        "score": score,  # ratio of tests passing (all-or-nothing here)
    }
    if error is not None:
        rec["error"] = error

    results.append(rec)

with open(output_jsonl, "w", encoding="utf-8") as f:
    for rec in results:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

mean_score = sum(r["score"] for r in results) / len(results)
print(f"Mean test pass ratio over HumanEval (all-or-nothing): {mean_score:.4f}")
# +
output_jsonl = "humaneval_qwen2_5_instruction_eval.jsonl"

results = []
with open(output_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

total = len(results)
num_pass = sum(1 for r in results if r.get("score", 0.0) == 1.0)
num_fail = total - num_pass
pass_ratio = num_pass / total if total > 0 else 0.0

print(f"Total tasks: {total}")
print(f"Passed: {num_pass}")
print(f"Failed: {num_fail}")
print(f"Pass@1 (test pass ratio): {pass_ratio:.4f}")

# -
