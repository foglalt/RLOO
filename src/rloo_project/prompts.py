"""Prompt templates and reward helpers for RLOO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class PromptTemplate:
    instruction: str

    def render(self, code_prefix: str) -> str:
        return self.instruction.format(code_prefix=code_prefix)


DEFAULT_TEMPLATE = PromptTemplate(
    instruction=(
        "You are an expert software engineer. Complete the following code snippet\n"
        "so that it forms a valid and idiomatic solution. Provide only the continuation."
        "\n\n{code_prefix}"
    )
)


def get_reward_fn(name: str) -> Callable[[str, str], float]:
    """Return a reward function that compares reference and generated code."""

    registry: Dict[str, Callable[[str, str], float]] = {
        "heuristic": _heuristic_reward,
        "lexical_overlap": _lexical_overlap_reward,
    }
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(f"Unknown reward function '{name}'. Options: {list(registry)}") from exc


def _heuristic_reward(reference: str, generated: str) -> float:
    """Simple non-differentiable proxy reward."""

    score = 0.0
    if reference.strip() and generated.strip():
        score += 0.3
    if reference.splitlines() and generated.splitlines():
        score += min(len(generated.splitlines()), len(reference.splitlines())) * 0.05
    if generated.endswith(":") or generated.endswith("{"):
        score -= 0.1
    return max(score, 0.0)


def _lexical_overlap_reward(reference: str, generated: str) -> float:
    import evaluate

    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=[generated], references=[reference])
    return float(result.get("rougeL", 0.0))


__all__ = ["PromptTemplate", "DEFAULT_TEMPLATE", "get_reward_fn"]
