#!/usr/bin/env python3
"""Count average generated response tokens from OPSD evaluation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_TARGET_STEPS = (25, 50, 75, 100)
DEFAULT_CONDITIONS = ("reference", "skeleton")
EVAL_FILE_RE = re.compile(
    r"^(?P<dataset>.+)_(?P<condition>reference|skeleton)_checkpoint_(?P<step>\d+)\.json$"
)
CHECKPOINT_RE = re.compile(r"(?:checkpoint[-_]?)(?P<step>\d+)")


@dataclass(frozen=True)
class EvalFileMetadata:
    dataset: str
    condition: str
    checkpoint_step: int


@dataclass(frozen=True)
class EvalFileSummary:
    dataset: str
    condition: str
    checkpoint_step: int
    path: Path
    response_count: int
    total_response_tokens: int
    average_response_tokens: float
    problem_count: int
    val_n: int | None
    average_at_n_pct: float | None
    pass_at_n_pct: float | None
    majority_vote_at_n_pct: float | None
    format_rate: float | None


def token_count(tokenizer: Any, text: str) -> int:
    """Tokenize generated text without adding BOS/EOS or other wrapper tokens."""
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    return len(token_ids)


def parse_eval_filename(path: Path) -> EvalFileMetadata | None:
    match = EVAL_FILE_RE.match(path.name)
    if not match:
        return None
    return EvalFileMetadata(
        dataset=match.group("dataset"),
        condition=match.group("condition"),
        checkpoint_step=int(match.group("step")),
    )


def _condition_from_stem(stem: str) -> str | None:
    for condition in DEFAULT_CONDITIONS:
        if re.search(rf"(^|[_-]){condition}([_-]|$)", stem):
            return condition
    return None


def _checkpoint_step_from_stem(stem: str) -> int | None:
    match = CHECKPOINT_RE.search(stem)
    if not match:
        return None
    return int(match.group("step"))


def infer_eval_metadata(path: Path, payload: dict[str, Any]) -> EvalFileMetadata:
    metadata = parse_eval_filename(path)
    if metadata is not None:
        return metadata

    dataset = str(payload.get("dataset") or path.stem)
    condition = _condition_from_stem(path.stem)
    checkpoint_step = _checkpoint_step_from_stem(path.stem)
    if condition is None or checkpoint_step is None:
        raise ValueError(
            f"{path}: could not parse dataset/condition/checkpoint from filename. "
            "Expected names like aime25_reference_checkpoint_50.json."
        )

    return EvalFileMetadata(
        dataset=dataset,
        condition=condition,
        checkpoint_step=checkpoint_step,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_texts_from_result(result: dict[str, Any], path: Path, result_index: int) -> Iterable[str]:
    generations = result.get("generations")
    if isinstance(generations, list):
        for generation_index, generation in enumerate(generations):
            if not isinstance(generation, dict):
                raise ValueError(
                    f"{path}: results[{result_index}].generations[{generation_index}] is not a JSON object"
                )
            full_generation = generation.get("full_generation")
            if not isinstance(full_generation, str):
                raise ValueError(
                    f"{path}: results[{result_index}].generations[{generation_index}] "
                    "missing string field 'full_generation'"
                )
            yield full_generation
        return

    full_generation = result.get("full_generation")
    if not isinstance(full_generation, str):
        raise ValueError(
            f"{path}: results[{result_index}] missing list field 'generations' "
            "and string field 'full_generation'"
        )
    yield full_generation


def extract_response_texts(payload: dict[str, Any], path: Path) -> tuple[list[str], int]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path}: missing list field 'results'")

    response_texts: list[str] = []
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"{path}: results[{result_index}] is not a JSON object")
        response_texts.extend(_response_texts_from_result(result, path, result_index))

    if not response_texts:
        raise ValueError(f"{path}: no response texts found under results")
    return response_texts, len(results)


def summarize_eval_file(path: Path, tokenizer: Any) -> EvalFileSummary:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    metadata = infer_eval_metadata(path, payload)
    response_texts, problem_count = extract_response_texts(payload, path)
    token_lengths = [token_count(tokenizer, text) for text in response_texts]
    total_response_tokens = sum(token_lengths)

    return EvalFileSummary(
        dataset=metadata.dataset,
        condition=metadata.condition,
        checkpoint_step=metadata.checkpoint_step,
        path=path,
        response_count=len(token_lengths),
        total_response_tokens=total_response_tokens,
        average_response_tokens=total_response_tokens / len(token_lengths),
        problem_count=problem_count,
        val_n=_optional_int(payload.get("val_n")),
        average_at_n_pct=_optional_float(payload.get("average_at_n_pct")),
        pass_at_n_pct=_optional_float(payload.get("pass_at_n_pct")),
        majority_vote_at_n_pct=_optional_float(payload.get("majority_vote_at_n_pct")),
        format_rate=_optional_float(payload.get("format_rate")),
    )


def normalize_target_steps(values: Sequence[int]) -> list[int]:
    steps = sorted(set(int(value) for value in values))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("Target steps must be positive integers")
    return steps


def _condition_order(condition: str) -> tuple[int, str]:
    if condition in DEFAULT_CONDITIONS:
        return (DEFAULT_CONDITIONS.index(condition), condition)
    return (len(DEFAULT_CONDITIONS), condition)


def _sort_key(path: Path) -> tuple[str, tuple[int, str], int, str]:
    metadata = parse_eval_filename(path)
    if metadata is None:
        return (path.stem, (len(DEFAULT_CONDITIONS), ""), 0, path.name)
    return (
        metadata.dataset,
        _condition_order(metadata.condition),
        metadata.checkpoint_step,
        path.name,
    )


def _candidate_files(inputs: Sequence[Path]) -> Iterable[tuple[Path, bool]]:
    for input_path in inputs:
        input_path = input_path.expanduser()
        if input_path.is_dir():
            for path in sorted(input_path.glob("*.json"), key=_sort_key):
                yield path, True
        elif input_path.is_file():
            yield input_path, False
        else:
            raise FileNotFoundError(f"Evaluation input does not exist: {input_path}")


def discover_eval_files(
    inputs: Sequence[Path],
    datasets: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    target_steps: Sequence[int] | None = None,
) -> list[Path]:
    dataset_filter = set(datasets or [])
    condition_filter = set(conditions or [])
    step_filter = set(target_steps or [])
    seen: set[Path] = set()
    files: list[Path] = []

    for path, from_directory in _candidate_files(inputs):
        if path.suffix != ".json":
            continue

        metadata = parse_eval_filename(path)
        if metadata is None and from_directory:
            continue
        if metadata is not None:
            if dataset_filter and metadata.dataset not in dataset_filter:
                continue
            if condition_filter and metadata.condition not in condition_filter:
                continue
            if step_filter and metadata.checkpoint_step not in step_filter:
                continue

        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            files.append(path)

    return sorted(files, key=_sort_key)


def ensure_unique_keys(summaries: Sequence[EvalFileSummary]) -> None:
    seen: dict[tuple[str, str, int], Path] = {}
    for summary in summaries:
        key = (summary.dataset, summary.condition, summary.checkpoint_step)
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Duplicate evaluation files for "
                f"dataset={summary.dataset}, condition={summary.condition}, "
                f"checkpoint={summary.checkpoint_step}: {previous} and {summary.path}"
            )
        seen[key] = summary.path


def rows_for_export(summaries: Sequence[EvalFileSummary]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sorted(
        summaries,
        key=lambda item: (item.dataset, _condition_order(item.condition), item.checkpoint_step),
    ):
        rows.append(
            {
                "dataset": summary.dataset,
                "condition": summary.condition,
                "checkpoint_step": summary.checkpoint_step,
                "eval_file": str(summary.path),
                "problem_count": summary.problem_count,
                "val_n": summary.val_n,
                "response_count": summary.response_count,
                "total_response_tokens": summary.total_response_tokens,
                "average_response_tokens": summary.average_response_tokens,
                "average_at_n_pct": summary.average_at_n_pct,
                "pass_at_n_pct": summary.pass_at_n_pct,
                "majority_vote_at_n_pct": summary.majority_vote_at_n_pct,
                "format_rate": summary.format_rate,
            }
        )
    return rows


def _format_float(value: float | None, precision: int) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def format_markdown_tables(
    summaries: Sequence[EvalFileSummary],
    target_steps: Sequence[int],
    precision: int = 1,
) -> str:
    by_key = {
        (summary.dataset, summary.condition, summary.checkpoint_step): summary
        for summary in summaries
    }
    datasets = sorted({summary.dataset for summary in summaries})
    conditions = sorted({summary.condition for summary in summaries}, key=_condition_order)

    blocks: list[str] = []
    for dataset in datasets:
        rows = [f"Dataset: {dataset}"]
        rows.append("| condition | " + " | ".join(str(step) for step in target_steps) + " |")
        rows.append("|---|" + "|".join("---" for _ in target_steps) + "|")
        for condition in conditions:
            values = []
            for step in target_steps:
                summary = by_key.get((dataset, condition, step))
                values.append(
                    _format_float(summary.average_response_tokens, precision)
                    if summary is not None
                    else ""
                )
            rows.append("| " + condition + " | " + " | ".join(values) + " |")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, ensure_ascii=False)


def load_tokenizer(tokenizer_name_or_path: str, trust_remote_code: bool, use_fast: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count average generated response token length from OPSD "
            "eval_results/*.json files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Evaluation result JSON files or directories containing eval result JSON files.",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or local path, usually the base model path used for evaluation.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset filter, e.g. aime25 or aime24 aime25 hmmt25.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=DEFAULT_CONDITIONS,
        help="Optional condition filter. Defaults to every condition found in the input files.",
    )
    parser.add_argument(
        "--target-steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_TARGET_STEPS),
        help="Checkpoint steps to include in the printed comparison table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Decimal places for average response token length in the printed table.",
    )
    parser.add_argument("--output-csv", type=Path, help="Optional path for detailed CSV output.")
    parser.add_argument("--output-json", type=Path, help="Optional path for detailed JSON output.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.from_pretrained.",
    )
    parser.add_argument(
        "--slow-tokenizer",
        action="store_true",
        help="Use the slow tokenizer implementation instead of use_fast=True.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    target_steps = normalize_target_steps(args.target_steps)
    files = discover_eval_files(
        args.inputs,
        datasets=args.datasets,
        conditions=args.conditions,
        target_steps=target_steps,
    )
    if not files:
        raise FileNotFoundError("No matching evaluation JSON files found.")

    tokenizer = load_tokenizer(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.slow_tokenizer,
    )
    summaries = [summarize_eval_file(path, tokenizer) for path in files]
    ensure_unique_keys(summaries)

    rows = rows_for_export(summaries)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, rows)

    print("Average generated response token length")
    print(format_markdown_tables(summaries, target_steps, precision=args.precision))

    if args.output_csv:
        print(f"\nCSV written to: {args.output_csv}")
    if args.output_json:
        print(f"JSON written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
