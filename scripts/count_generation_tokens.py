#!/usr/bin/env python3
"""Count cumulative generated tokens from OPSD generation JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TARGET_STEPS = (25, 50, 75, 100)
GENERATION_FILE_RE = re.compile(r"^generations_step_(\d+)\.json$")


@dataclass(frozen=True)
class GenerationFileStats:
    step: int
    path: Path
    generated_tokens: int
    samples: int


@dataclass(frozen=True)
class RunSummary:
    label: str
    run_path: Path
    generations_dir: Path
    file_stats: tuple[GenerationFileStats, ...]
    tokens_by_target: dict[int, int]
    samples_by_target: dict[int, int]
    files_by_target: dict[int, int]


def parse_run_spec(run_spec: str) -> tuple[str, Path]:
    """Parse either PATH or label=PATH into a display label and filesystem path."""
    if "=" in run_spec:
        label, path_text = run_spec.split("=", 1)
        if not label:
            raise ValueError(f"Run spec has an empty label: {run_spec!r}")
        if not path_text:
            raise ValueError(f"Run spec has an empty path: {run_spec!r}")
        return label, Path(path_text).expanduser()

    path = Path(run_spec).expanduser()
    label = path.parent.name if path.name == "generations" else path.name
    return label or str(path), path


def resolve_generations_dir(run_path: Path) -> Path:
    """Accept an experiment directory or its nested generations directory."""
    run_path = run_path.expanduser()
    if run_path.is_dir() and any(run_path.glob("generations_step_*.json")):
        return run_path

    generations_dir = run_path / "generations"
    if generations_dir.is_dir():
        return generations_dir

    raise FileNotFoundError(
        f"Could not find generation JSON files under {run_path}. "
        "Pass either the experiment directory or its generations/ directory."
    )


def generation_step_from_path(path: Path) -> int:
    match = GENERATION_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Generation file name does not match generations_step_<N>.json: {path}")
    return int(match.group(1))


def token_count(tokenizer: Any, text: str) -> int:
    """Tokenize generated text without adding BOS/EOS or other wrapper tokens."""
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    return len(token_ids)


def generation_record_token_count(record: Any, tokenizer: Any, path: Path, index: int) -> int:
    if not isinstance(record, dict):
        raise ValueError(f"{path}: generation record #{index} is not a JSON object")

    completion = record.get("completion")
    if not isinstance(completion, str):
        raise ValueError(f"{path}: generation record #{index} missing string field 'completion'")

    return token_count(tokenizer, completion)


def read_generation_file(path: Path, tokenizer: Any) -> GenerationFileStats:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")

    step = generation_step_from_path(path)
    payload_step = payload.get("step")
    if isinstance(payload_step, int) and payload_step != step:
        raise ValueError(f"{path}: file step {step} does not match top-level step {payload_step}")

    generations = payload.get("generations")
    if not isinstance(generations, list):
        raise ValueError(f"{path}: missing list field 'generations'")

    generated_tokens = sum(
        generation_record_token_count(record, tokenizer, path, index)
        for index, record in enumerate(generations)
    )
    return GenerationFileStats(
        step=step,
        path=path,
        generated_tokens=generated_tokens,
        samples=len(generations),
    )


def discover_generation_files(generations_dir: Path) -> list[Path]:
    files = sorted(generations_dir.glob("generations_step_*.json"), key=generation_step_from_path)
    if not files:
        raise FileNotFoundError(f"No generations_step_*.json files found in {generations_dir}")
    return files


def cumulative_stats(
    file_stats: Sequence[GenerationFileStats], target_steps: Sequence[int]
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    tokens_by_target: dict[int, int] = {}
    samples_by_target: dict[int, int] = {}
    files_by_target: dict[int, int] = {}

    sorted_stats = sorted(file_stats, key=lambda stats: stats.step)
    for target_step in target_steps:
        included = [stats for stats in sorted_stats if stats.step <= target_step]
        tokens_by_target[target_step] = sum(stats.generated_tokens for stats in included)
        samples_by_target[target_step] = sum(stats.samples for stats in included)
        files_by_target[target_step] = len(included)

    return tokens_by_target, samples_by_target, files_by_target


def summarize_run(
    label: str,
    run_path: Path,
    tokenizer: Any,
    target_steps: Sequence[int] = DEFAULT_TARGET_STEPS,
) -> RunSummary:
    generations_dir = resolve_generations_dir(run_path)
    file_stats = tuple(read_generation_file(path, tokenizer) for path in discover_generation_files(generations_dir))
    tokens_by_target, samples_by_target, files_by_target = cumulative_stats(file_stats, target_steps)

    return RunSummary(
        label=label,
        run_path=run_path,
        generations_dir=generations_dir,
        file_stats=file_stats,
        tokens_by_target=tokens_by_target,
        samples_by_target=samples_by_target,
        files_by_target=files_by_target,
    )


def normalize_target_steps(values: Sequence[int]) -> list[int]:
    steps = sorted(set(int(value) for value in values))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("Target steps must be positive integers")
    return steps


def format_token_value(value: int, scale: str, precision: int) -> str:
    if scale == "raw":
        return str(value)
    if scale == "million":
        return f"{value / 1_000_000:.{precision}f}"
    raise ValueError(f"Unsupported scale: {scale}")


def format_markdown_table(
    summaries: Sequence[RunSummary],
    target_steps: Sequence[int],
    scale: str = "raw",
    precision: int = 3,
) -> str:
    header = "| run | " + " | ".join(str(step) for step in target_steps) + " |"
    separator = "|---|" + "|".join("---" for _ in target_steps) + "|"
    rows = [header, separator]
    for summary in summaries:
        values = [
            format_token_value(summary.tokens_by_target[step], scale=scale, precision=precision)
            for step in target_steps
        ]
        rows.append("| " + summary.label + " | " + " | ".join(values) + " |")
    return "\n".join(rows)


def rows_for_export(summaries: Sequence[RunSummary], target_steps: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        available_steps = {stats.step for stats in summary.file_stats}
        for target_step in target_steps:
            rows.append(
                {
                    "run": summary.label,
                    "generations_dir": str(summary.generations_dir),
                    "target_step": target_step,
                    "cumulative_generated_tokens": summary.tokens_by_target[target_step],
                    "cumulative_samples": summary.samples_by_target[target_step],
                    "cumulative_files": summary.files_by_target[target_step],
                    "has_exact_generation_file": target_step in available_steps,
                }
            )
    return rows


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
            "Count cumulative generated completion tokens from OPSD "
            "generations/generations_step_<N>.json files."
        )
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Experiment directories or generations directories. Use label=PATH "
            "to control the row label, e.g. reference=/data/run_ref skeleton=/data/run_skel."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or local path, usually the base model path used for training.",
    )
    parser.add_argument(
        "--target-steps",
        nargs="+",
        type=int,
        default=list(DEFAULT_TARGET_STEPS),
        help="Training steps at which to report cumulative generated tokens.",
    )
    parser.add_argument(
        "--scale",
        choices=("raw", "million"),
        default="raw",
        help="Scale used in the printed Markdown table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places when --scale million is used.",
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
    tokenizer = load_tokenizer(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.slow_tokenizer,
    )

    summaries = []
    for run_spec in args.runs:
        label, run_path = parse_run_spec(run_spec)
        summary = summarize_run(label=label, run_path=run_path, tokenizer=tokenizer, target_steps=target_steps)
        summaries.append(summary)

        available_steps = {stats.step for stats in summary.file_stats}
        for target_step in target_steps:
            if target_step not in available_steps:
                print(
                    f"warning: {summary.label} has no generations_step_{target_step}.json; "
                    f"reported value uses files with step <= {target_step}",
                    file=sys.stderr,
                )

    rows = rows_for_export(summaries, target_steps)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, rows)

    scale_note = "raw tokens" if args.scale == "raw" else f"millions of tokens (10^6), {args.precision} decimals"
    print(f"Cumulative generated completion tokens ({scale_note})")
    print(format_markdown_table(summaries, target_steps, scale=args.scale, precision=args.precision))

    if args.output_csv:
        print(f"\nCSV written to: {args.output_csv}")
    if args.output_json:
        print(f"JSON written to: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
