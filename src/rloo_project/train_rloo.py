"""Entry point to run RLOO fine-tuning for code completion."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from datasets import Dataset
from transformers import GenerationConfig, set_seed
from yamlargparse import ArgumentParser

from .config import TrainingConfig
from .data import get_source_column, load_code_dataset
from .modeling import PolicyComponents, load_policy_model
from .prompts import DEFAULT_TEMPLATE, get_reward_fn

try:
    from trl import RLOOConfig, RLOOTrainer
except ImportError as exc:  # pragma: no cover - handled in README instructions
    raise SystemExit(
        "The 'trl' package is required to run training. Install dependencies via requirements.txt"
    ) from exc


def run_training(config: TrainingConfig) -> None:
    set_seed(config.seed)

    dataset = load_code_dataset(config.dataset)
    source_column = get_source_column(dataset)
    processed_dataset = _build_prompt_dataset(
        dataset,
        source_column,
        max_examples=config.dataset.sample_size,
    )

    policy = load_policy_model(config.model, config.prompt)
    reference_model = _load_reference_model(config)

    trl_config = _build_trl_config(config)

    reward_impl = get_reward_fn(config.reward.shaping)

    def reward_fn(samples: List[str], prompts: List[str], originals: List[Dict[str, Any]]):
        rewards: List[float] = []
        for sample, prompt, original in zip(samples, prompts, originals):
            reference = original["reference_response"]
            rewards.append(reward_impl(reference, sample))
        return torch.tensor(rewards, dtype=torch.float32)

    trainer = RLOOTrainer(
        config=trl_config,
        model=policy.model,
        ref_model=reference_model,
        tokenizer=policy.tokenizer,
        reward_fn=reward_fn,
        train_dataset=processed_dataset,
        generation_config=_build_generation_config(config),
    )

    trainer.train()
    trainer.model.save_pretrained(config.run.output_dir)
    trainer.tokenizer.save_pretrained(config.run.output_dir)


def _build_prompt_dataset(
    dataset: Dataset | Iterable[dict],
    source_column: str,
    max_examples: int | None = None,
) -> Dataset:
    records: List[Dict[str, Any]] = []

    for sample in dataset:
        code = sample[source_column]
        prefix, suffix = _split_code_example(code)
        rendered = DEFAULT_TEMPLATE.render(code_prefix=prefix)
        records.append({
            "prompt": rendered,
            "reference_response": suffix,
        })

        if max_examples is not None and len(records) >= max_examples:
            break

    if not records:
        raise ValueError("No samples were collected from the dataset; check configuration filters.")

    return Dataset.from_list(records)


def _split_code_example(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    if len(lines) < 2:
        midpoint = max(1, len(text) // 2)
        return text[:midpoint], text[midpoint:]

    midpoint = max(1, int(len(lines) * 0.6))
    prefix = "\n".join(lines[:midpoint])
    suffix = "\n".join(lines[midpoint:])
    return prefix, suffix


def _load_reference_model(config: TrainingConfig):
    model_cfg = deepcopy(config.model)
    model_cfg.use_peft_lora = False
    components = load_policy_model(model_cfg, config.prompt)
    # Freeze parameters of the reference model
    for param in components.model.parameters():
        param.requires_grad = False
    return components.model


def _build_trl_config(config: TrainingConfig) -> RLOOConfig:
    return RLOOConfig(
        learning_rate=config.optimizer.learning_rate,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=config.run.gradient_accumulation_steps,
        max_steps=config.run.max_steps,
    )


def _build_generation_config(config: TrainingConfig) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=config.prompt.max_new_tokens,
        temperature=config.inference.temperature,
        top_p=config.inference.top_p,
        num_return_sequences=1,
        eos_token_id=None,
        pad_token_id=None,
    )


def _parse_cli_args() -> TrainingConfig:
    parser = ArgumentParser(description="Train a small LM with RLOO for code completion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="Literal JSON string with parameter overrides (optional)",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)
    if args.overrides:
        overrides = json.loads(args.overrides)
        cfg = TrainingConfig.from_dict(_merge_dicts(cfg.to_dict(), overrides))
    return cfg


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and key in out:
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    training_config = _parse_cli_args()
    Path(training_config.run.output_dir).mkdir(parents=True, exist_ok=True)
    run_training(training_config)
