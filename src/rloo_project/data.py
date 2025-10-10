"""Dataset utilities for RLOO code completion training."""

from __future__ import annotations

from typing import Iterable, Optional

from datasets import Dataset, load_dataset

from .config import DatasetConfig


_LANG_COLUMN_CANDIDATES = ("lang", "language", "repo_language")
_SOURCE_COLUMN_CANDIDATES = ("content", "code", "text")


def load_code_dataset(cfg: DatasetConfig) -> Dataset | Iterable[dict]:
    """Load a dataset according to the configuration.

    When `streaming` is True, returns an iterable dataset.
    """

    ds = load_dataset(
        path=cfg.name,
        split=cfg.split,
        streaming=cfg.streaming,
        use_auth_token=False,
    )

    if cfg.language_filter:
        ds = _filter_by_language(ds, cfg.language_filter)

    if not cfg.streaming and cfg.sample_size:
        return ds.select(range(min(cfg.sample_size, len(ds))))

    if cfg.streaming and cfg.sample_size:
        return list(_take_samples(ds, cfg.sample_size))

    return ds


def get_source_column(dataset: Dataset | Iterable[dict]) -> str:
    """Infer the code column name from dataset features."""

    if isinstance(dataset, Dataset):
        for column in _SOURCE_COLUMN_CANDIDATES:
            if column in dataset.column_names:
                return column
        raise KeyError(
            f"Unable to infer source column from {dataset.column_names}."
        )

    # Generic iterable/list case
    iterator = iter(dataset)
    try:
        first = next(iterator)
    except StopIteration as exc:  # pragma: no cover - handled via caller
        raise ValueError("Dataset iterable is empty; cannot infer source column.") from exc

    for column in _SOURCE_COLUMN_CANDIDATES:
        if column in first:
            return column
    raise KeyError(f"Unable to infer source column from sample keys {list(first.keys())}.")


def _filter_by_language(dataset: Dataset | Iterable[dict], language: str):
    if isinstance(dataset, Dataset):
        column = _infer_language_column(dataset.column_names)
        if column:
            return dataset.filter(lambda row: row.get(column) == language)
        return dataset

    # Streaming filter
    column = None
    iterator = iter(dataset)
    for sample in iterator:
        if column is None:
            column = _infer_language_column(sample.keys())
        if column is None or sample.get(column) == language:
            yield sample


def _take_samples(dataset: Iterable[dict], sample_size: int):
    count = 0
    for sample in dataset:
        yield sample
        count += 1
        if count >= sample_size:
            break


def _infer_language_column(columns: Iterable[str]) -> Optional[str]:
    for candidate in _LANG_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


__all__ = [
    "DatasetConfig",
    "load_code_dataset",
    "get_source_column",
]
