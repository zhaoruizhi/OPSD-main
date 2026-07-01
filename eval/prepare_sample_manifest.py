#!/usr/bin/env python
"""Prepare fixed sample-index manifests for OPSD quick experiments."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from .quick_opsd_common import choose_stratified_indices, read_jsonl, write_json
except ImportError:  # pragma: no cover
    from quick_opsd_common import choose_stratified_indices, read_jsonl, write_json


def extract_indices_from_rollouts(records: list[dict[str, Any]], condition: str) -> list[int]:
    indices = {
        int(record["problem_id"])
        for record in records
        if record.get("condition") == condition and "problem_id" in record
    }
    return sorted(indices)


def build_manifest(
    dataset: str,
    split: str,
    sample_size: int,
    seed: int,
    indices: list[int],
) -> dict[str, Any]:
    sorted_indices = sorted({int(index) for index in indices})
    return {
        "dataset": dataset,
        "split": split,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "indices": sorted_indices,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed sample-index manifest.")
    parser.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--from-rollout-file",
        type=str,
        help="Existing rollouts.jsonl to extract problem_id values from.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="student",
        help="Rollout condition to extract when --from-rollout-file is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_rollout_file:
        records = read_jsonl(args.from_rollout_file)
        indices = extract_indices_from_rollouts(records, args.condition)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, split=args.split)
        rows = [dict(row) for row in dataset]
        indices = choose_stratified_indices(rows, args.sample_size, args.seed)

    manifest = build_manifest(
        dataset=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        indices=indices,
    )
    write_json(args.output_file, manifest)


if __name__ == "__main__":
    main()
