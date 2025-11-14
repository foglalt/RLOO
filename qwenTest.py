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

prompt_start = 'You are an expert Python coding assistant.\nFollow these rules when solving the task below:\n- Implement the requested function exactly once using the provided signature.\n- Return efficient, idiomatic Python 3 code.\n- Do not include markdown, explanations, tests, or extra helper text—only executable code.\n'
prompt_end = '\ndef count_upper(s):\n    """\n    Given a string s, count the number of uppercase vowels in even indices.\n    \n    For example:\n    count_upper(\'aBCdEf\') returns 1\n    count_upper(\'abcdefg\') returns 0\n    count_upper(\'dBBE\') returns 0\n    """'

response = qwen_coder_chat(prompt_start+prompt_end)
print(response)
# -


