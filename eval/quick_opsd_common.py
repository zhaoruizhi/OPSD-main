"""Shared utilities for OPSD quick-run probes.

The quick-run scripts are intentionally isolated from training. Heavy runtime
dependencies such as vLLM and Transformers are imported by the entrypoint
scripts only when needed so this module stays cheap to unit test.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STYLE_MARKERS = {
    "wait",
    "think",
    "thinking",
    "actually",
    "maybe",
    "hmm",
    "oops",
    "instead",
    "reconsider",
    "restart",
}

MATH_MARKERS = {
    "\\frac",
    "\\sqrt",
    "\\boxed",
    "\\cdot",
    "\\times",
    "\\leq",
    "\\geq",
    "\\sum",
    "\\int",
    "=",
    "+",
    "-",
    "*",
    "/",
    "^",
    "<",
    ">",
}

RESTART_PATTERNS = [
    r"\blet'?s start over\b",
    r"\bstart from scratch\b",
    r"\bsolve from scratch\b",
    r"\brestart\b",
    r"\binstead,?\s+(?:we|i)\s+(?:solve|will solve|should solve)",
    r"\bignore (?:the|this) previous\b",
]


def extract_boxed_answer(text: str | None) -> str | None:
    """Extract the last ``\\boxed{...}`` answer, including nested braces."""
    if not text:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    brace_start = text.find("{", idx)
    if brace_start < 0:
        return None

    depth = 0
    for pos in range(brace_start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos].strip()
    return None


def _normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    return re.sub(r"[\s$]", "", str(answer)).lower().strip()


def grade_answer(predicted: str | None, ground_truth: str | None) -> bool:
    """Grade with math_verify when available, otherwise use normalized text."""
    if predicted is None or ground_truth is None:
        return False

    try:
        from math_verify import parse, verify

        pred_text = predicted if "$" in predicted else f"${predicted}$"
        gt_text = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        pred_parsed = parse(pred_text, fallback_mode="no_fallback")
        gt_parsed = parse(gt_text, fallback_mode="no_fallback")
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        return _normalize_answer(predicted) == _normalize_answer(ground_truth)


def get_ground_truth_answer(example: dict[str, Any]) -> str | None:
    """Return the best available final answer from an OPSD/OpenThoughts row."""
    for key in ("Answer", "answer", "ground_truth", "final_answer"):
        value = example.get(key)
        if value not in (None, ""):
            return str(value)

    solution = example.get("solution") or example.get("COT_Reason")
    boxed = extract_boxed_answer(str(solution) if solution is not None else None)
    return boxed or (str(solution) if solution not in (None, "") else None)


def get_problem_text(example: dict[str, Any]) -> str:
    for key in ("problem", "Question", "question", "prompt"):
        value = example.get(key)
        if value not in (None, ""):
            return str(value)
    raise KeyError("Could not find problem text in example")


def get_solution_text(example: dict[str, Any]) -> str:
    for key in ("solution", "COT_Reason", "reasoning"):
        value = example.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def generated_token_count(example: dict[str, Any]) -> int:
    value = example.get("generated_token_count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def choose_stratified_indices(rows: Iterable[dict[str, Any]], sample_size: int, seed: int) -> list[int]:
    """Deterministically sample short/mid/long reasoning examples."""
    rows_list = list(rows)
    total = len(rows_list)
    if sample_size <= 0:
        return []
    if sample_size >= total:
        return list(range(total))

    rng = random.Random(seed)
    ordered = sorted(range(total), key=lambda idx: (generated_token_count(rows_list[idx]), idx))
    bucket_size = math.ceil(total / 3)
    buckets = [ordered[i * bucket_size : (i + 1) * bucket_size] for i in range(3)]

    base = sample_size // 3
    quotas = [base, base, base]
    for i in range(sample_size % 3):
        quotas[i] += 1

    selected: list[int] = []
    for bucket, quota in zip(buckets, quotas):
        take = min(quota, len(bucket))
        selected.extend(rng.sample(bucket, take))

    if len(selected) < sample_size:
        remaining = [idx for idx in range(total) if idx not in set(selected)]
        selected.extend(rng.sample(remaining, sample_size - len(selected)))

    return sorted(selected)


def shard_items(items: list[Any], shard_id: int, num_shards: int) -> list[Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return [item for pos, item in enumerate(items) if pos % num_shards == shard_id]


def build_student_user_message(problem: str) -> str:
    return f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def _answer_line(answer: str | None) -> str:
    if answer in (None, ""):
        return ""
    return f"\nFinal answer: {answer}\n"


def build_reference_user_message(problem: str, solution: str, answer: str | None = None) -> str:
    transition_prompt = (
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step - do not copy or paraphrase it. Now, using your "
        "own words and independent reasoning, derive the same final answer to the problem above. "
        "Think step by step, explore different approaches, and don't be afraid to backtrack "
        "or reconsider if something doesn't work out:\n"
    )
    return (
        f"Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{_answer_line(answer)}"
        f"{transition_prompt}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def build_opsd_oracle_user_message(problem: str, solution: str) -> str:
    return build_reference_user_message(problem, solution)


def normalize_semantic_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(skeleton, dict):
        raise ValueError("semantic skeleton must be a JSON object")

    critical_intermediates = skeleton.get("critical_intermediates")
    if critical_intermediates is None:
        critical_intermediates = skeleton.get("critical_intermediate", [])

    checks = skeleton.get("checks")
    if checks is None:
        checks = skeleton.get("check", [])

    return {
        "final_answer": skeleton.get("final_answer"),
        "key_objects": _list_or_empty(skeleton.get("key_objects")),
        "subgoals": _list_or_empty(skeleton.get("subgoals")),
        "critical_intermediates": _list_or_empty(critical_intermediates),
        "theorem_tags": _list_or_empty(skeleton.get("theorem_tags")),
        "checks": _list_or_empty(checks),
    }


def build_semantic_skeleton_user_message(
    problem: str,
    skeleton: dict[str, Any],
    answer: str | None = None,
) -> str:
    normalized_skeleton = normalize_semantic_skeleton(skeleton)
    final_answer = answer if answer not in (None, "") else normalized_skeleton.get("final_answer")
    skeleton_without_answer = {
        key: value for key, value in normalized_skeleton.items() if key != "final_answer"
    }
    skeleton_json = json.dumps(
        skeleton_without_answer,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return build_reference_user_message(problem, skeleton_json, answer=final_answer)


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_heuristic_diagnostic(ground_truth: str | None) -> dict[str, str]:
    answer_hint = ground_truth if ground_truth not in (None, "") else "the verified final answer"
    return {
        "validity": "uncertain",
        "first_invalid_span": "not identified by heuristic quick run",
        "local_reason": "Use the privileged signal only to check local mathematical validity.",
        "minimal_repair_hint": f"Continue toward the verified final answer {answer_hint} without copying a reference solution.",
        "next_local_subgoal": "Complete the immediate next algebraic or logical step from the existing prefix.",
    }


def build_intervention_user_message(problem: str, diagnostic: dict[str, Any]) -> str:
    diagnostic_json = json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
    return (
        f"Problem:\n{problem}\n\n"
        "Hidden diagnostic information, not a target style:\n"
        f"{diagnostic_json}\n\n"
        "Rules:\n"
        "- Do not restart from the reference solution.\n"
        "- Do not copy the reference wording.\n"
        "- Continue exactly from the existing assistant prefix.\n"
        "- Preserve the student's plan, notation, and variable names unless they are mathematically invalid.\n"
        "- If the prefix is valid, preserve the student's plan and notation.\n"
        "- If the prefix is invalid, make the smallest local repair needed and continue.\n"
        "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )


def render_chat_prompt(tokenizer: Any, user_message: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def render_prefill_prompt(
    tokenizer: Any,
    user_message: str,
    assistant_prefix: str,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_prefix},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = render_chat_prompt(tokenizer, user_message, enable_thinking=enable_thinking)
        return prompt + assistant_prefix


def split_prefix_by_token_ratio(tokenizer: Any, text: str, ratio: float = 0.5) -> tuple[str, str, int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if not token_ids:
        return "", "", 0

    clipped_ratio = min(max(ratio, 0.0), 1.0)
    cutoff = int(len(token_ids) * clipped_ratio)
    cutoff = min(max(cutoff, 1), len(token_ids))
    prefix = tokenizer.decode(token_ids[:cutoff], skip_special_tokens=True)
    tail = tokenizer.decode(token_ids[cutoff:], skip_special_tokens=True)
    return prefix, tail, cutoff


def detect_restart(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in RESTART_PATTERNS)


def _ngram_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return re.findall(r"\\[a-zA-Z]+|\w+|[^\s]", text.lower())


def ngram_overlap_rate(candidate: str | None, reference: str | None, n: int = 4) -> float:
    cand_tokens = _ngram_tokens(candidate)
    ref_tokens = _ngram_tokens(reference)
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0

    cand_ngrams = Counter(tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum((cand_ngrams & ref_ngrams).values())
    return overlap / max(1, sum(cand_ngrams.values()))


def is_style_token(token_text: str) -> bool:
    lowered = token_text.strip().lower()
    if not lowered:
        return False
    return lowered in STYLE_MARKERS or any(marker in lowered for marker in ("start over", "from scratch"))


def is_math_token(token_text: str) -> bool:
    if any(char.isdigit() for char in token_text):
        return True
    return any(marker in token_text for marker in MATH_MARKERS)


def _looks_like_new_solution(continuation: str | None) -> bool:
    if not continuation:
        return False
    stripped = continuation.lstrip().lower()
    return stripped.startswith(("problem:", "solution:", "we need to solve", "let us solve"))


def _notation_consistency(prefix: str, continuation: str, restart: bool) -> float:
    if restart:
        return 0.0

    prefix_terms = set(re.findall(r"(?<![A-Za-z])[a-zA-Z](?![A-Za-z])", prefix))
    continuation_terms = set(re.findall(r"(?<![A-Za-z])[a-zA-Z](?![A-Za-z])", continuation))
    if not prefix_terms or not continuation_terms:
        return 1.0
    return len(prefix_terms & continuation_terms) / len(prefix_terms)


def continuation_metrics(
    prefix: str,
    continuation: str,
    ground_truth: str | None,
    reference_solution: str | None,
) -> dict[str, Any]:
    full_text = f"{prefix}{continuation}"
    predicted = extract_boxed_answer(full_text)
    restart = detect_restart(continuation)
    reference_copy_rate = ngram_overlap_rate(continuation, reference_solution, n=4)
    prefix_preserved = not restart and not _looks_like_new_solution(continuation)
    notation_consistency = _notation_consistency(prefix, continuation, restart)
    locality_score = max(0.0, (1.0 - min(reference_copy_rate, 1.0)) * (1.0 if prefix_preserved else 0.0))
    return {
        "predicted_answer": predicted,
        "formatted": predicted is not None,
        "correct": grade_answer(predicted, ground_truth),
        "restart": restart,
        "prefix_preserved": prefix_preserved,
        "notation_consistency": notation_consistency,
        "locality_score": locality_score,
        "reference_copy_rate": reference_copy_rate,
        "completion_chars": len(continuation or ""),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_sample_indices_file(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    indices = payload.get("indices") if isinstance(payload, dict) else payload
    if not isinstance(indices, list):
        raise ValueError("sample indices file must contain an indices list")
    return [int(index) for index in indices]


def read_skeleton_file(path: str | Path) -> dict[int, dict[str, Any]]:
    skeletons: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        status = str(record.get("status", "ok")).lower()
        if status not in {"ok", "success"}:
            continue
        if "problem_id" not in record:
            raise ValueError("skeleton record is missing problem_id")
        skeleton = record.get("skeleton")
        if isinstance(skeleton, str):
            skeleton = json.loads(skeleton)
        skeletons[int(record["problem_id"])] = normalize_semantic_skeleton(skeleton)
    return skeletons


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def summarize_generation_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("condition", "unknown")].append(record)

    summary: dict[str, Any] = {"conditions": {}}
    for condition, condition_records in sorted(grouped.items()):
        by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in condition_records:
            by_problem[str(record.get("problem_id"))].append(record)

        pass_count = sum(any(item.get("correct") for item in items) for items in by_problem.values())
        majority_count = 0
        for items in by_problem.values():
            formatted = [item.get("predicted_answer") for item in items if item.get("formatted")]
            if not formatted:
                continue
            majority_answer = Counter(formatted).most_common(1)[0][0]
            if grade_answer(majority_answer, items[0].get("ground_truth")):
                majority_count += 1

        total = len(condition_records)
        problem_count = len(by_problem)
        correct_count = sum(1 for record in condition_records if record.get("correct"))
        formatted_count = sum(1 for record in condition_records if record.get("formatted"))
        restart_count = sum(1 for record in condition_records if record.get("restart"))
        token_lengths = [
            record.get("completion_tokens")
            for record in condition_records
            if isinstance(record.get("completion_tokens"), int)
        ]

        summary["conditions"][condition] = {
            "num_problems": problem_count,
            "total_generations": total,
            "avg_at_n": correct_count / total if total else 0.0,
            "pass_at_n": pass_count / problem_count if problem_count else 0.0,
            "majority_vote": majority_count / problem_count if problem_count else 0.0,
            "format_rate": formatted_count / total if total else 0.0,
            "restart_rate": restart_count / total if total else 0.0,
            "avg_completion_tokens": sum(token_lengths) / len(token_lengths) if token_lengths else 0.0,
            "avg_reference_copy_rate": _mean(
                record.get("reference_copy_rate") for record in condition_records
            ),
            "prefix_preservation_rate": _mean(
                1.0 if record.get("prefix_preserved") else 0.0 for record in condition_records
            ),
            "avg_notation_consistency": _mean(
                record.get("notation_consistency") for record in condition_records
            ),
            "avg_locality_score": _mean(record.get("locality_score") for record in condition_records),
        }

    return summary


def summarize_logit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    contrast_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entropy_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record_type = record.get("record_type")
        if record_type == "rollout_entropy":
            entropy_records[str(record.get("condition", "unknown"))].append(record)
        elif record.get("contrast") is not None:
            contrast_records[str(record.get("contrast", "unknown"))].append(record)

    return {
        "contrasts": {
            contrast: {
                "num_cases": len(items),
                "mean_kl": _mean(item.get("mean_kl") for item in items),
                "mean_top1_agreement": _mean(item.get("top1_agreement") for item in items),
                "mean_topk_jaccard": _mean(item.get("topk_jaccard") for item in items),
                "mean_delta_logp_target": _mean(item.get("mean_delta_logp_target") for item in items),
                "mean_style_kl_share": _mean(item.get("style_kl_share") for item in items),
                "mean_math_kl_share": _mean(item.get("math_kl_share") for item in items),
                "mean_first_window_kl_share": _mean(item.get("first_window_kl_share") for item in items),
                "mean_teacher_entropy": _mean(item.get("mean_teacher_entropy") for item in items),
                "mean_student_entropy": _mean(item.get("mean_student_entropy") for item in items),
                "mean_delta_entropy": _mean(item.get("mean_delta_entropy") for item in items),
            }
            for contrast, items in sorted(contrast_records.items())
        },
        "rollout_entropy": {
            condition: {
                "num_cases": len(items),
                "mean_entropy": _mean(item.get("mean_entropy") for item in items),
            }
            for condition, items in sorted(entropy_records.items())
        },
    }


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0
