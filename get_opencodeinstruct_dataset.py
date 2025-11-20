"""Download the first 10,000 rows from the OpenCodeInstruct dataset.

This version relies on the Hugging Face ``datasets`` library instead of the
public dataset-server API. Authenticate first via ``huggingface-cli login`` to
guarantee access before running the script.
"""

from __future__ import annotations

import json
import re
from itertools import islice
from typing import Iterable

from datasets import load_dataset


DATASET = "nvidia/OpenCodeInstruct"
CONFIG = "train"
SPLIT = "train"
ROWS_TO_DOWNLOAD = 10_000
OUTPUT_PATH = "dataset.jsonl"
FUNC_NAME_RE = re.compile(r"assert\s+([a-zA-Z_]\w*)\s*\(")


def parse_unit_tests(raw_tests: str | None) -> list[str] | str | None:
	if raw_tests is None:
		return None
	try:
		parsed = json.loads(raw_tests)
		if isinstance(parsed, list):
			return parsed
		return raw_tests
	except json.JSONDecodeError:
		return raw_tests


def _unit_tests_iterable(unit_tests: list[str] | str | None) -> Iterable[str]:
	if unit_tests is None:
		return ()
	if isinstance(unit_tests, list):
		return (test for test in unit_tests if isinstance(test, str))
	if isinstance(unit_tests, str):
		return (unit_tests,)
	return ()


def extract_entry_point(unit_tests: list[str] | str | None) -> str | None:
	for test in _unit_tests_iterable(unit_tests):
		match = FUNC_NAME_RE.search(test)
		if match:
			return match.group(1)
	return None


def iter_dataset_rows(limit: int) -> Iterable[dict]:
	dataset = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
	return islice(dataset, limit)


def main() -> None:
	count = 0
	with open(OUTPUT_PATH, "w", encoding="utf-8") as sink:
		for idx, row in enumerate(iter_dataset_rows(ROWS_TO_DOWNLOAD)):
			parsed_tests = parse_unit_tests(row.get("unit_tests"))
			row_idx = row.get("row_idx", idx)
			record = {
				"id": row_idx,
				"input": row.get("input"),
				"unit_tests": parsed_tests,
				"entry_point": extract_entry_point(parsed_tests),
			}
			sink.write(json.dumps(record, ensure_ascii=False) + "\n")
			count += 1

	print(f"Saved {count} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
