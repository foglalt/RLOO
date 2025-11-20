"""Download slices of the OpenCodeInstruct dataset via Hugging Face ``datasets``.

Authenticate first via ``huggingface-cli login`` to guarantee access before
running the script. You can control which slice to export using CLI flags.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import islice
from typing import Iterable

from datasets import load_dataset


DATASET = "nvidia/OpenCodeInstruct"
CONFIG = "train"
SPLIT = "train"
DEFAULT_ROWS_STARTFROM = 0
DEFAULT_ROWS_TO_DOWNLOAD = 10_000
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


def iter_dataset_rows(start: int, limit: int) -> Iterable[dict]:
	dataset = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)
	return islice(dataset, start, start + limit)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export slices of OpenCodeInstruct")
	parser.add_argument(
		"--rows-startfrom",
		type=int,
		default=DEFAULT_ROWS_STARTFROM,
		help="Zero-based row index to start exporting from (default: %(default)s)",
	)
	parser.add_argument(
		"--rows-to-download",
		type=int,
		default=DEFAULT_ROWS_TO_DOWNLOAD,
		help="Number of rows to export (default: %(default)s)",
	)
	return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)
	if args.rows_startfrom < 0:
		raise ValueError("rows-startfrom must be non-negative")
	if args.rows_to_download <= 0:
		raise ValueError("rows-to-download must be positive")

	output_path = f"opencodeinstruct_{args.rows_startfrom}_{args.rows_startfrom + args.rows_to_download}.jsonl"
	count = 0
	with open(output_path, "w", encoding="utf-8") as sink:
		for fallback_idx, row in enumerate(iter_dataset_rows(args.rows_startfrom, args.rows_to_download)):
			parsed_tests = parse_unit_tests(row.get("unit_tests"))
			row_idx = row.get("row_idx", args.rows_startfrom + fallback_idx)
			record = {
				"id": row_idx,
				"input": row.get("input"),
				"unit_tests": parsed_tests,
				"entry_point": extract_entry_point(parsed_tests),
			}
			sink.write(json.dumps(record, ensure_ascii=False) + "\n")
			count += 1

	print(f"Saved {count} rows to {output_path}")


if __name__ == "__main__":
	main()
