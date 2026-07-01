#!/usr/bin/env python
"""Generate DeepSeek intervention diagnostics for sampled OPSD rollouts."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SAMPLE_INDICES = SCRIPT_DIR / "data" / "sample_indices.json"
DEFAULT_ROLLOUT_FILE = (
    WORK_ROOT
    / "Privileged Context"
    / "OPSD-main"
    / "outputs"
    / "opsd_quick"
    / "qwen31b_skeleton_ablation_reuse_20260629_112333"
    / "rollouts.jsonl"
)
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "outputs" / "deepseek_v4_pro_intervention_diagnostics.jsonl"

PROBLEM_FIELDS = ("problem", "Question", "question", "prompt")
SOLUTION_FIELDS = ("solution", "COT_Reason", "reasoning")
DIAGNOSTIC_FIELDS = (
    "prefix_valid_until",
    "first_error_span",
    "error_type",
    "valid_prefix_summary",
    "student_plan",
    "local_repair",
    "next_subgoal_after_repair",
)
ALLOWED_ERROR_TYPES = {
    "none",
    "arithmetic_error",
    "algebraic_error",
    "invalid_implication",
    "unjustified_assumption",
    "missing_case",
    "theorem_misuse",
    "definition_misuse",
    "notation_confusion",
    "contradiction_ignored",
    "answer_mismatch",
    "incomplete_solution",
    "other",
    "uncertain",
}

JUDGE_SYSTEM_PROMPT = """You are a local intervention judge.

Given a problem, reference solution, and a student's numbered reasoning trace, produce a diagnostic object.

Your goal is to identify the earliest point where the teacher should intervene.

The intervention must be:
- student-aware;
- prefix-preserving;
- minimal;
- local;
- mathematically grounded;

Judging procedure:

1. Read the problem and reference solution.
2. Inspect the student's trace step by step.
3. Determine the longest mathematically valid student prefix.
4. Locate the first invalid or unsafe step.
5. Describe the student's plan before that point.
6. Provide the smallest repair that allows the student to continue from their own prefix.
7. Provide the next local subgoal after the repair.

Return exactly one JSON object:

{
  "prefix_valid_until": string,
  "first_error_span": string or null,
  "error_type": string,
  "valid_prefix_summary": string,
  "student_plan": string,
  "local_repair": string,
  "next_subgoal_after_repair": string
}

Allowed error_type values:
"none",
"arithmetic_error",
"algebraic_error",
"invalid_implication",
"unjustified_assumption",
"missing_case",
"theorem_misuse",
"definition_misuse",
"notation_confusion",
"contradiction_ignored",
"answer_mismatch",
"incomplete_solution",
"other",
"uncertain".

Output constraints:

- Output JSON only.
- local_repair must be at most 80 words.
- next_subgoal_after_repair must be at most 40 words.
- If no error is found:
  - first_error_span = null
  - error_type = "none"
  - local_repair = ""
  - next_subgoal_after_repair = null.
- If the first error is uncertain:
  - error_type = "uncertain"
  - first_error_span should point to the earliest uncertain step
  - local_repair should state the missing check or condition."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-indices", type=Path, default=DEFAULT_SAMPLE_INDICES)
    parser.add_argument("--rollout-file", type=Path, default=DEFAULT_ROLLOUT_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default="DeepSeek-v4-pro")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--limit", type=int, help="Only process the first N sampled problems.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def first_present(row: dict[str, Any], field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return str(value)
    raise KeyError(f"Could not find any of fields: {', '.join(field_names)}")


def load_sample_manifest(path: Path) -> tuple[str, str, list[int]]:
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected object in {path}")

    dataset_name = manifest.get("dataset")
    split = manifest.get("split")
    indices = manifest.get("indices")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"{path} must contain a non-empty string field: dataset")
    if not isinstance(split, str) or not split:
        raise ValueError(f"{path} must contain a non-empty string field: split")
    if not isinstance(indices, list) or not all(isinstance(item, int) for item in indices):
        raise ValueError(f"{path} must contain an integer list field: indices")
    return dataset_name, split, indices


def load_dataset_examples(dataset_name: str, split: str, indices: list[int]) -> dict[int, dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Run with the conda tool environment.") from exc

    dataset = load_dataset(dataset_name, split=split)
    missing = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
    if missing:
        raise ValueError(f"Sample indices outside dataset range: {missing[:20]}")

    examples: dict[int, dict[str, str]] = {}
    for idx in indices:
        row = dict(dataset[idx])
        examples[idx] = {
            "problem": first_present(row, PROBLEM_FIELDS),
            "solution": first_present(row, SOLUTION_FIELDS),
        }
    return examples


def load_student_rollouts(path: Path, indices: list[int]) -> dict[int, str]:
    wanted = set(indices)
    rollouts: dict[int, str] = {}
    duplicates: list[int] = []

    for record in read_jsonl(path):
        if record.get("condition") != "student" or record.get("sample_index") != 0:
            continue
        try:
            problem_id = int(record["problem_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Rollout record is missing an integer problem_id: {record}") from exc
        if problem_id not in wanted:
            continue
        if problem_id in rollouts:
            duplicates.append(problem_id)
            continue
        full_generation = record.get("full_generation")
        if not isinstance(full_generation, str) or not full_generation.strip():
            raise ValueError(f"Missing full_generation for problem_id={problem_id}")
        rollouts[problem_id] = full_generation

    missing = [idx for idx in indices if idx not in rollouts]
    if missing:
        raise ValueError(f"Missing student sample_index=0 rollouts for problem ids: {missing}")
    if duplicates:
        raise ValueError(f"Duplicate student sample_index=0 rollouts for problem ids: {sorted(set(duplicates))}")
    return rollouts


def numbered_reasoning_trace(full_generation: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", full_generation.strip()) if block.strip()]
    return "\n\n".join(f"{idx}. {block}" for idx, block in enumerate(blocks, start=1))


def build_user_message(problem: str, solution: str, student_trace: str) -> str:
    return (
        "Problem:\n"
        f"{problem}\n\n"
        "Reference solution:\n"
        f"{solution}\n\n"
        "Student numbered reasoning trace:\n"
        f"{student_trace}"
    )


def word_count(value: Any) -> int:
    if value is None:
        return 0
    return len(str(value).split())


def normalize_model_name(model: str) -> str:
    aliases = {
        "DeepSeek-v4-pro": "deepseek-v4-pro",
        "DeepSeek-v4-flash": "deepseek-v4-flash",
    }
    return aliases.get(model, model)


def validate_diagnostic(value: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return ["diagnostic is not a JSON object"]

    keys = set(value)
    expected = set(DIAGNOSTIC_FIELDS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"extra fields: {', '.join(extra)}")
    if errors:
        return errors

    for field_name in (
        "prefix_valid_until",
        "error_type",
        "valid_prefix_summary",
        "student_plan",
        "local_repair",
    ):
        if not isinstance(value[field_name], str):
            errors.append(f"{field_name} must be a string")

    if value["first_error_span"] is not None and not isinstance(value["first_error_span"], str):
        errors.append("first_error_span must be a string or null")
    next_subgoal = value["next_subgoal_after_repair"]
    if next_subgoal is not None and not isinstance(next_subgoal, str):
        errors.append("next_subgoal_after_repair must be a string or null")

    error_type = value["error_type"]
    if error_type not in ALLOWED_ERROR_TYPES:
        errors.append(f"error_type is not allowed: {error_type}")

    if word_count(value["local_repair"]) > 80:
        errors.append("local_repair exceeds 80 words")
    if word_count(next_subgoal) > 40:
        errors.append("next_subgoal_after_repair exceeds 40 words")

    if error_type == "none":
        if value["first_error_span"] is not None:
            errors.append('first_error_span must be null when error_type is "none"')
        if value["local_repair"] != "":
            errors.append('local_repair must be empty when error_type is "none"')
        if next_subgoal is not None:
            errors.append('next_subgoal_after_repair must be null when error_type is "none"')
    return errors


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output was valid JSON but not an object")
    return value


def chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    json_mode: bool,
) -> str:
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("Missing dependency: requests. Run with the conda tool environment.") from exc

    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": normalize_model_name(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if not content.strip():
                finish_reason = choice.get("finish_reason")
                raise RuntimeError(
                    f"DeepSeek returned empty content (finish_reason={finish_reason}); "
                    "try increasing --max-tokens"
                )
            return str(content)
        except Exception as exc:  # noqa: BLE001 - preserve API error context across retries.
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"DeepSeek request failed after {retries} attempts: {last_error}")


def request_diagnostic(
    *,
    api_base: str,
    api_key: str,
    model: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int,
    json_mode: bool,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    raw_response = chat_completion(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        json_mode=json_mode,
    )
    try:
        diagnostic = parse_json_object(raw_response)
    except Exception as exc:  # noqa: BLE001 - repair prompt includes the parse failure.
        validation_errors = [f"JSON parse failed: {exc}"]
    else:
        validation_errors = validate_diagnostic(diagnostic)
        if not validation_errors:
            return diagnostic

    repair_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "The previous response did not satisfy the required JSON schema.\n"
                f"Validation errors: {json.dumps(validation_errors, ensure_ascii=False)}\n\n"
                "Return one corrected JSON object only. Do not include Markdown or commentary.\n\n"
                f"Original task:\n{user_message}\n\n"
                f"Previous response:\n{raw_response}"
            ),
        },
    ]
    repaired = chat_completion(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=repair_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        json_mode=json_mode,
    )
    diagnostic = parse_json_object(repaired)
    validation_errors = validate_diagnostic(diagnostic)
    if validation_errors:
        raise ValueError(f"Repaired diagnostic is still invalid: {validation_errors}")
    return diagnostic


def load_completed_problem_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    for record in read_jsonl(path):
        problem_id = record.get("problem_id")
        diagnostic = record.get("diagnostic")
        if isinstance(problem_id, int) and not validate_diagnostic(diagnostic):
            completed.add(problem_id)
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key: set {args.api_key_env}")

    dataset_name, split, indices = load_sample_manifest(args.sample_indices)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        indices = indices[: args.limit]

    examples = load_dataset_examples(dataset_name, split, indices)
    rollouts = load_student_rollouts(args.rollout_file, indices)
    completed = load_completed_problem_ids(args.output_file) if args.resume else set()

    if not args.resume and args.output_file.exists():
        args.output_file.unlink()

    print(
        f"Loaded {len(indices)} sampled problems from {dataset_name}/{split}; "
        f"{len(completed)} already completed.",
        flush=True,
    )

    for ordinal, problem_id in enumerate(indices, start=1):
        if problem_id in completed:
            print(f"[{ordinal}/{len(indices)}] skip problem_id={problem_id} (already completed)", flush=True)
            continue

        example = examples[problem_id]
        student_trace = numbered_reasoning_trace(rollouts[problem_id])
        user_message = build_user_message(example["problem"], example["solution"], student_trace)
        print(f"[{ordinal}/{len(indices)}] judging problem_id={problem_id}", flush=True)
        diagnostic = request_diagnostic(
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            user_message=user_message,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            json_mode=args.json_mode,
        )
        append_jsonl(
            args.output_file,
            {
                "problem_id": problem_id,
                "diagnostic": diagnostic,
            },
        )

    print(f"Wrote diagnostics to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
