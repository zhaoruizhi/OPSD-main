#!/usr/bin/env python
"""Validate and atomically merge JSONL shard files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield validated JSON objects and report the exact corrupt source line."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {source}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            yield record


def _write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_jsonl_files(
    input_paths: list[str | Path],
    output_path: str | Path,
    sort_key: str | None = None,
) -> int:
    """Merge JSONL files without exposing a partial aggregate on failure."""

    if not input_paths:
        raise ValueError("At least one input JSONL file is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    count = 0
    try:
        if sort_key is None:
            for input_path in input_paths:
                for record in iter_jsonl_records(input_path):
                    _write_record(temp_handle, record)
                    count += 1
        else:
            records = [
                record
                for input_path in input_paths
                for record in iter_jsonl_records(input_path)
            ]
            missing = [record for record in records if sort_key not in record]
            if missing:
                raise ValueError(f"Sort key {sort_key!r} is missing from {len(missing)} record(s)")
            records.sort(key=lambda record: record[sort_key])
            for record in records:
                _write_record(temp_handle, record)
            count = len(records)

        temp_handle.flush()
        os.fsync(temp_handle.fileno())
        temp_handle.close()
        os.replace(temp_path, output)
    except BaseException:
        if not temp_handle.closed:
            temp_handle.close()
        temp_path.unlink(missing_ok=True)
        raise

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and atomically merge JSONL shard files.")
    parser.add_argument(
        "--input-file",
        action="append",
        required=True,
        help="Input JSONL file. Repeat in the desired merge order.",
    )
    parser.add_argument("--output-file", required=True, help="Atomic merged JSONL output path.")
    parser.add_argument(
        "--sort-key",
        help="Optional record key used to sort small merged outputs, for example 'rank'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = merge_jsonl_files(args.input_file, args.output_file, sort_key=args.sort_key)
    print(json.dumps({"output_file": args.output_file, "num_records": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
