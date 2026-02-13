"""Download slices of the OpenCodeInstruct dataset via Hugging Face ``datasets``.

Authenticate first via ``huggingface-cli login`` to guarantee access before
running the script. You can control which slice to export using CLI flags.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
from collections import Counter
from itertools import islice
from typing import Iterable

from datasets import load_dataset


DATASET = "nvidia/OpenCodeInstruct"
CONFIG = "train"
SPLIT = "train"
DEFAULT_ROWS_STARTFROM = 0
DEFAULT_ROWS_TO_DOWNLOAD = 10_000
ASSERT_CALL_RE = re.compile(r"assert\s+([a-zA-Z_]\w*)\s*\(")
PROMPT_NAME_RE = re.compile(r"(?:function|method|class)\s+`([a-zA-Z_]\w*)`", re.IGNORECASE)
PROMPT_CALL_RE = re.compile(r"`([a-zA-Z_]\w*)\s*\(")
PROMPT_DEF_RE = re.compile(r"\bdef\s+([a-zA-Z_]\w*)\s*\(")
DISALLOWED_ENTRY_POINTS = set(dir(builtins)) | {"check", "candidate"}
UNSUPPORTED_TEST_PATTERNS = [
	("input_required", re.compile(r"\binput\s*\(", re.IGNORECASE)),
	("file_io", re.compile(r"\b(open|os\.|pathlib\.|shutil|tempfile|glob|walk)\b", re.IGNORECASE)),
	("network", re.compile(r"\b(requests|urllib|httpx|socket|flask|fastapi|aiohttp|http\.client)\b", re.IGNORECASE)),
	("database", re.compile(r"\b(sqlite3|sqlalchemy|pymongo|psycopg|mysql|postgres)\b", re.IGNORECASE)),
	("system_monitoring", re.compile(r"\bpsutil\b", re.IGNORECASE)),
	("heavy_deps", re.compile(r"\b(matplotlib|wordcloud|pandas|numpy|sklearn|nltk|cv2|PIL|seaborn)\b", re.IGNORECASE)),
	("subprocess", re.compile(r"\bsubprocess\b", re.IGNORECASE)),
]
UNSUPPORTED_PROMPT_PATTERNS = [
	("file_or_dir_task", re.compile(r"\b(file|files|directory|directories|filesystem|path|folder|os\.walk|walk through)\b", re.IGNORECASE)),
	("api_or_server_task", re.compile(r"\b(restful|flask|fastapi|endpoint|api server|web server)\b", re.IGNORECASE)),
	("database_task", re.compile(r"\b(database|sqlite|sql|mongodb|postgres)\b", re.IGNORECASE)),
]


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


def normalize_unit_tests(raw_tests: str | None) -> list[str]:
	parsed = parse_unit_tests(raw_tests)
	if isinstance(parsed, list):
		return [test.strip() for test in parsed if isinstance(test, str) and test.strip()]
	if isinstance(parsed, str) and parsed.strip():
		return [parsed.strip()]
	return []


def _unit_tests_iterable(unit_tests: list[str] | str | None) -> Iterable[str]:
	if unit_tests is None:
		return ()
	if isinstance(unit_tests, list):
		return (test for test in unit_tests if isinstance(test, str))
	if isinstance(unit_tests, str):
		return (unit_tests,)
	return ()


def _iter_called_names(test_src: str) -> Iterable[str]:
	try:
		tree = ast.parse(test_src)
	except SyntaxError:
		return ()

	names: list[str] = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
			names.append(node.func.id)
	return names


def _iter_prompt_candidates(prompt: str | None) -> Iterable[str]:
	if not prompt:
		return ()

	names: list[str] = []
	for pattern in (PROMPT_NAME_RE, PROMPT_CALL_RE, PROMPT_DEF_RE):
		names.extend(pattern.findall(prompt))
	return names


def extract_entry_point(unit_tests: list[str] | str | None, prompt: str | None = None) -> str | None:
	"""Infer a likely callable under test from tests and task prompt text."""
	counts: Counter[str] = Counter()
	ordered_candidates: list[str] = []

	def add(name: str, weight: int = 1) -> None:
		if name not in counts:
			ordered_candidates.append(name)
		counts[name] += weight

	for test in _unit_tests_iterable(unit_tests):
		for name in ASSERT_CALL_RE.findall(test):
			add(name, weight=4)
		for name in _iter_called_names(test):
			add(name, weight=1)

	for name in _iter_prompt_candidates(prompt):
		add(name, weight=2)

	order = {name: idx for idx, name in enumerate(ordered_candidates)}
	scored = sorted(ordered_candidates, key=lambda name: (-counts[name], order[name]))
	for name in scored:
		if name in DISALLOWED_ENTRY_POINTS:
			continue
		return name
	return None


def detect_unsupported_task(instruction: str | None, unit_tests: list[str]) -> list[str]:
	reasons: list[str] = []
	tests_blob = "\n".join(unit_tests)
	prompt = instruction or ""
	for label, pattern in UNSUPPORTED_TEST_PATTERNS:
		if pattern.search(tests_blob):
			reasons.append(f"test:{label}")
	for label, pattern in UNSUPPORTED_PROMPT_PATTERNS:
		if pattern.search(prompt):
			reasons.append(f"prompt:{label}")
	return reasons


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
	skipped_missing_entry_point = 0
	skipped_unsupported_task = 0
	unsupported_reason_counts: Counter[str] = Counter()
	with open(output_path, "w", encoding="utf-8") as sink:
		for fallback_idx, row in enumerate(iter_dataset_rows(args.rows_startfrom, args.rows_to_download)):
			instruction = row.get("input")
			parsed_tests = normalize_unit_tests(row.get("unit_tests"))

			entry_point = extract_entry_point(parsed_tests, instruction)
			if entry_point is None:
				skipped_missing_entry_point += 1
				continue
			unsupported_reasons = detect_unsupported_task(instruction, parsed_tests)
			if unsupported_reasons:
				skipped_unsupported_task += 1
				for reason in unsupported_reasons:
					unsupported_reason_counts[reason] += 1
				continue

			row_idx = row.get("row_idx", args.rows_startfrom + fallback_idx)
			record = {
				"id": row_idx,
				"input": instruction,
				"unit_tests": parsed_tests,
				"entry_point": entry_point,
			}
			sink.write(json.dumps(record, ensure_ascii=False) + "\n")
			count += 1

	print(f"Saved {count} rows to {output_path}")
	if skipped_missing_entry_point:
		print(f"Skipped {skipped_missing_entry_point} rows: missing_entry_point")
	if skipped_unsupported_task:
		reason_summary = ", ".join(f"{reason}={amount}" for reason, amount in unsupported_reason_counts.most_common())
		print(f"Skipped {skipped_unsupported_task} rows: unsupported_task ({reason_summary})")


if __name__ == "__main__":
	main()
