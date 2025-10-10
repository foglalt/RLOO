"""Configuration dataclasses and helpers for RLOO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass
class RunConfig:
    output_dir: str = "/kaggle/working/rloo_runs/default"
    logging_steps: int = 5
    save_steps: int = 50
    eval_strategy: str = "steps"
    eval_steps: int = 50
    max_steps: int = 200
    gradient_accumulation_steps: int = 1
    mixed_precision: Optional[str] = None


@dataclass
class ModelConfig:
    name_or_path: str = "Salesforce/codegen-350M-mono"
    revision: Optional[str] = None
    trust_remote_code: bool = False
    use_peft_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class RewardConfig:
    shaping: str = "heuristic"
    baseline: str = "running_mean"
    clip_range: float = 5.0


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass
class PromptConfig:
    max_source_length: int = 256
    max_new_tokens: int = 128
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n"])


@dataclass
class DatasetConfig:
    name: str = "codeparrot/codeparrot-clean"
    split: str = "train"
    streaming: bool = True
    sample_size: Optional[int] = 1000
    language_filter: Optional[str] = "python"


@dataclass
class InferenceConfig:
    num_return_sequences: int = 4
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class TrainingConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        data = _load_yaml(path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            run=RunConfig(**data.get("run", {})),
            model=ModelConfig(**data.get("model", {})),
            reward=RewardConfig(**data.get("reward", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            prompt=PromptConfig(**data.get("prompt", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            seed=data.get("seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": _asdict(self.run),
            "model": _asdict(self.model),
            "reward": _asdict(self.reward),
            "optimizer": _asdict(self.optimizer),
            "prompt": _asdict(self.prompt),
            "dataset": _asdict(self.dataset),
            "inference": _asdict(self.inference),
            "seed": self.seed,
        }


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _asdict(instance: Any) -> Dict[str, Any]:
    if hasattr(instance, "__dict__"):
        return {k: _normalize(v) for k, v in instance.__dict__.items()}
    raise TypeError(f"Object {instance!r} is not dataclass-like")


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize(v) for v in value)
    if hasattr(value, "__dict__"):
        return _asdict(value)
    return value
