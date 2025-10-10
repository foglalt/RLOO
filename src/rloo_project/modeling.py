"""Model helpers for RLOO code completion fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig, PromptConfig


@dataclass
class PolicyComponents:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_policy_model(model_cfg: ModelConfig, prompt_cfg: PromptConfig) -> PolicyComponents:
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name_or_path,
        revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name_or_path,
        revision=model_cfg.revision,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    if model_cfg.use_peft_lora:
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=None,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.use_cache = False

    return PolicyComponents(model=model, tokenizer=tokenizer)
