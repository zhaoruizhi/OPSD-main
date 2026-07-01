#!/usr/bin/env python
"""Logit distribution probe for OPSD quick runs."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .quick_opsd_common import (
        build_heuristic_diagnostic,
        build_intervention_user_message,
        build_opsd_oracle_user_message,
        build_reference_user_message,
        build_semantic_skeleton_user_message,
        build_student_user_message,
        is_math_token,
        is_style_token,
        read_jsonl,
        read_skeleton_file,
        render_chat_prompt,
        render_prefill_prompt,
        shard_items,
        summarize_logit_records,
        write_json,
    )
except ImportError:  # pragma: no cover
    from quick_opsd_common import (
        build_heuristic_diagnostic,
        build_intervention_user_message,
        build_opsd_oracle_user_message,
        build_reference_user_message,
        build_semantic_skeleton_user_message,
        build_student_user_message,
        is_math_token,
        is_style_token,
        read_jsonl,
        read_skeleton_file,
        render_chat_prompt,
        render_prefill_prompt,
        shard_items,
        summarize_logit_records,
        write_json,
    )


LOGPROB_BACKEND = "hf_causal_lm"
TARGET_TOKEN_SOURCE_TEXT = "target_tail_text"
TARGET_TOKEN_SOURCE_TOKEN_IDS = "completion_token_ids"
PROMPT_TOKEN_SOURCE_TEXT = "reconstructed_prompt_text"
PROMPT_TOKEN_SOURCE_TOKEN_IDS = "prompt_token_ids"
TARGET_TOKEN_SOURCE = TARGET_TOKEN_SOURCE_TEXT
TENSOR_REDUCTION_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class LogitContextSpec:
    name: str
    enable_thinking: bool
    prompt_kind: str


def build_logit_context_specs() -> list[LogitContextSpec]:
    return [
        LogitContextSpec("student", enable_thinking=False, prompt_kind="student"),
        LogitContextSpec("teacher_base", enable_thinking=True, prompt_kind="base"),
        LogitContextSpec("teacher_reference", enable_thinking=True, prompt_kind="reference"),
        LogitContextSpec("teacher_skeleton", enable_thinking=True, prompt_kind="skeleton"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPSD quick logits distribution probe.")
    parser.add_argument("--model", type=str, default="/data0/shared/Qwen3-1.7B")
    parser.add_argument("--rollout-file", type=str)
    parser.add_argument("--prefix-file", type=str)
    parser.add_argument("--skeleton-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--summary-file", type=str, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--input-file", type=str, help="JSONL file to summarize when --summarize-only is set.")
    parser.add_argument("--logit-size", type=int, default=64)
    parser.add_argument(
        "--trajectory-condition",
        action="append",
        choices=["student", "teacher_base", "teacher_reference", "teacher_skeleton"],
        help="Full rollout condition to use as a target trajectory. Repeatable.",
    )
    parser.add_argument("--probe-tokens", type=int, default=0)
    parser.add_argument(
        "--trajectory-sample-index",
        type=int,
        default=0,
        help="Rollout sample_index to use for full-response KL/entropy. Use -1 to keep all samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--top-kl-positions", type=int, default=20)
    parser.add_argument("--first-window-tokens", type=int, default=32)
    parser.add_argument("--max-context-tokens", type=int, default=20000)
    parser.add_argument(
        "--skip-rollout-entropy",
        action="store_true",
        help=(
            "Only compute teacher-vs-base KL contrast records. The contrast records already include "
            "teacher/base entropy and target-logp deltas along the base trajectory."
        ),
    )
    parser.add_argument(
        "--hf-device-map",
        choices=["cuda", "auto", "cpu"],
        default="cuda",
        help=(
            "How to place the HuggingFace probe model. 'cuda' keeps the whole model on the visible GPU; "
            "'auto' allows Accelerate device_map auto placement/offload; 'cpu' forces CPU."
        ),
    )
    args = parser.parse_args()
    if args.trajectory_condition is None:
        args.trajectory_condition = ["teacher_base"]
    if args.summarize_only:
        if not args.input_file:
            parser.error("--input-file is required with --summarize-only")
        return args
    if not args.rollout_file and not args.prefix_file:
        parser.error("one of --rollout-file or --prefix-file is required")
    if not args.output_file:
        parser.error("--output-file is required unless --summarize-only is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        records = read_jsonl(args.input_file)
        write_json(args.summary_file, summarize_logit_records(records))
        return

    run_logit_probe(args)
    write_json(args.summary_file, summarize_logit_records(read_jsonl(args.output_file)))


def shard_cases(cases: list[dict[str, Any]], shard_id: int, num_shards: int) -> list[dict[str, Any]]:
    return shard_items(cases, shard_id, num_shards)


def logit_record_key(record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type") or "")
    case_id = str(record.get("case_id") or "")
    if record_type == "rollout_entropy":
        return f"rollout_entropy:{case_id}:{record.get('condition')}"
    contrast = record.get("contrast")
    if record_type == "kl_contrast" or contrast is not None:
        return f"kl_contrast:{case_id}:{contrast}"
    return f"{record_type}:{case_id}:{record.get('condition') or contrast}"


def completed_logit_record_keys(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    output_path = Path(path)
    if not output_path.exists():
        return set()
    return {logit_record_key(record) for record in read_jsonl(output_path)}


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _coerce_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(token_id) for token_id in value]
    except TypeError:
        return []


def select_logit_cases(records: list[dict[str, Any]], logit_size: int, seed: int) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("condition") != "c0_student_continue":
            continue
        case_id = str(record.get("case_id"))
        if case_id and case_id not in by_case and record.get("target_tail_text"):
            by_case[case_id] = record

    cases = list(by_case.values())
    if logit_size > 0 and len(cases) > logit_size:
        rng = random.Random(seed)
        cases = rng.sample(cases, logit_size)
    return sorted(cases, key=lambda item: str(item.get("case_id")))


def select_full_response_cases(
    records: list[dict[str, Any]],
    logit_size: int,
    seed: int,
    trajectory_conditions: list[str],
    trajectory_sample_index: int | None = None,
    context_conditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    required_conditions = set(trajectory_conditions)
    if context_conditions is not None:
        required_conditions.update(context_conditions)
    for record in records:
        condition = str(record.get("condition") or "")
        if condition not in required_conditions:
            continue
        if not record.get("full_generation") and not _coerce_token_ids(record.get("completion_token_ids")):
            continue
        problem_id = str(record.get("problem_id"))
        try:
            sample_index = int(record.get("sample_index", 0))
        except (TypeError, ValueError):
            sample_index = 0
        if trajectory_sample_index is not None and sample_index != trajectory_sample_index:
            continue
        grouped.setdefault((problem_id, sample_index), {})[condition] = record

    complete_groups = [
        (key, by_condition)
        for key, by_condition in grouped.items()
        if all(condition in by_condition for condition in required_conditions)
    ]
    complete_groups.sort(key=lambda item: (item[0][0], item[0][1]))
    if logit_size > 0 and len(complete_groups) > logit_size:
        rng = random.Random(seed)
        complete_groups = rng.sample(complete_groups, logit_size)
        complete_groups.sort(key=lambda item: (item[0][0], item[0][1]))

    cases: list[dict[str, Any]] = []
    for (_, _), by_condition in complete_groups:
        for condition in trajectory_conditions:
            record = by_condition[condition]
            try:
                sample_index = int(record.get("sample_index", 0))
            except (TypeError, ValueError):
                sample_index = 0
            problem_id = record.get("problem_id")
            full_generation = str(record.get("full_generation") or "")
            attached_contexts = (
                {condition: by_condition[condition] for condition in context_conditions}
                if context_conditions is not None
                else {}
            )
            cases.append(
                {
                    **record,
                    "case_id": f"{problem_id}:{sample_index}:{condition}",
                    "student_prefix": "",
                    "target_tail_text": full_generation,
                    "prefix_token_cutoff": 0,
                    "prefix_ratio": 0.0,
                    "target_condition": condition,
                    "target_sample_index": sample_index,
                    "context_records": attached_contexts,
                    "diagnostic": record.get("diagnostic") or build_heuristic_diagnostic(record.get("ground_truth")),
                }
            )
    return cases


def truncate_target_text(tokenizer: Any, target_text: str, probe_tokens: int) -> tuple[str, list[int]]:
    token_ids = tokenizer(str(target_text or ""), add_special_tokens=False)["input_ids"]
    if probe_tokens > 0:
        token_ids = token_ids[:probe_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=False), [int(token_id) for token_id in token_ids]


def target_token_ids_for_case(
    tokenizer: Any,
    case: dict[str, Any],
    probe_tokens: int,
) -> tuple[str, list[int], str]:
    token_ids = _coerce_token_ids(case.get("completion_token_ids"))
    if token_ids:
        if probe_tokens > 0:
            token_ids = token_ids[:probe_tokens]
        target_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        return target_text, token_ids, TARGET_TOKEN_SOURCE_TOKEN_IDS

    target_text, token_ids = truncate_target_text(
        tokenizer,
        str(case.get("target_tail_text") or ""),
        probe_tokens,
    )
    return target_text, token_ids, TARGET_TOKEN_SOURCE_TEXT


def context_prompt_ids_for_condition(
    tokenizer: Any,
    case: dict[str, Any],
    condition: str,
    skeletons: dict[int, dict[str, Any]],
) -> tuple[list[int], str]:
    context_records = case.get("context_records")
    record = context_records.get(condition, case) if isinstance(context_records, dict) else case
    prompt_ids = _coerce_token_ids(record.get("prompt_token_ids"))
    if prompt_ids:
        return prompt_ids, PROMPT_TOKEN_SOURCE_TOKEN_IDS

    prompt_text = rollout_context_prompt(
        tokenizer=tokenizer,
        case=case,
        condition=condition,
        skeletons=skeletons,
    )
    encoded = tokenizer(prompt_text, add_special_tokens=False)
    return _coerce_token_ids(encoded.get("input_ids")), PROMPT_TOKEN_SOURCE_TEXT


def run_logit_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.hf_device_map == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--hf-device-map cuda requires CUDA, but torch.cuda.is_available() is False")
        model_kwargs["device_map"] = {"": "cuda:0"}
    elif args.hf_device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    first_param_device = str(next(model.parameters()).device)
    print(
        json.dumps(
            {
                "event": "hf_probe_model_loaded",
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "dtype": str(dtype),
                "hf_device_map_arg": args.hf_device_map,
                "hf_device_map": getattr(model, "hf_device_map", None),
                "first_parameter_device": first_param_device,
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )

    skeletons = read_skeleton_file(args.skeleton_file) if args.skeleton_file else {}
    completed_keys = completed_logit_record_keys(args.output_file)
    output_records: list[dict[str, Any]] = []

    if args.rollout_file:
        rollout_records = read_jsonl(args.rollout_file)
        trajectory_sample_index = args.trajectory_sample_index if args.trajectory_sample_index >= 0 else None
        cases = select_full_response_cases(
            rollout_records,
            args.logit_size,
            args.seed,
            list(args.trajectory_condition),
            trajectory_sample_index=trajectory_sample_index,
            context_conditions=["teacher_base", "teacher_reference", "teacher_skeleton"],
        )
        cases = shard_cases(cases, args.shard_id, args.num_shards)
        for case in cases:
            contrast_specs = [
                ("teacher_reference_vs_teacher_base", "teacher_base", "teacher_reference"),
                ("teacher_skeleton_vs_teacher_base", "teacher_base", "teacher_skeleton"),
            ]
            pending = [
                item
                for item in contrast_specs
                if f"kl_contrast:{case.get('case_id')}:{item[0]}" not in completed_keys
            ]
            if not pending:
                continue

            _target_text, target_ids, target_token_source = target_token_ids_for_case(
                tokenizer,
                case,
                args.probe_tokens,
            )
            if not target_ids:
                continue

            condition_log_probs: dict[str, Any] = {}
            for _, base_condition, teacher_condition in pending:
                for condition in (base_condition, teacher_condition):
                    if condition in condition_log_probs:
                        continue
                    prompt_ids, _prompt_token_source = context_prompt_ids_for_condition(
                        tokenizer=tokenizer,
                        case=case,
                        condition=condition,
                        skeletons=skeletons,
                    )
                    condition_log_probs[condition] = compute_target_log_probs_hf(
                        model=model,
                        prompt_ids=prompt_ids,
                        target_ids=target_ids,
                        max_context_tokens=args.max_context_tokens,
                    )

            for contrast, base_condition, teacher_condition in pending:
                record = compare_contexts(
                    case=case,
                    tokenizer=tokenizer,
                    target_ids=target_ids,
                    student_log_probs=condition_log_probs[base_condition],
                    teacher_log_probs=condition_log_probs[teacher_condition],
                    contrast=contrast,
                    top_k=args.top_k,
                    top_kl_positions=args.top_kl_positions,
                    first_window_tokens=args.first_window_tokens,
                    target_token_source=target_token_source,
                )
                key = logit_record_key(record)
                if key in completed_keys:
                    continue
                append_jsonl_record(args.output_file, record)
                completed_keys.add(key)
                output_records.append(record)

        if not args.skip_rollout_entropy:
            entropy_cases = select_full_response_cases(
                rollout_records,
                args.logit_size,
                args.seed,
                ["student", "teacher_base", "teacher_reference", "teacher_skeleton"],
                trajectory_sample_index=trajectory_sample_index,
            )
            entropy_cases = shard_cases(entropy_cases, args.shard_id, args.num_shards)
            for case in entropy_cases:
                entropy_key = f"rollout_entropy:{case.get('case_id')}:{case.get('target_condition')}"
                if entropy_key in completed_keys:
                    continue
                _target_text, target_ids, target_token_source = target_token_ids_for_case(
                    tokenizer,
                    case,
                    args.probe_tokens,
                )
                if not target_ids:
                    continue
                condition = str(case.get("target_condition") or case.get("condition"))
                prompt_ids, _prompt_token_source = context_prompt_ids_for_condition(
                    tokenizer=tokenizer,
                    case=case,
                    condition=condition,
                    skeletons=skeletons,
                )
                log_probs = compute_target_log_probs_hf(
                    model=model,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    max_context_tokens=args.max_context_tokens,
                )
                record = build_rollout_entropy_record(
                    case=case,
                    tokenizer=tokenizer,
                    target_ids=target_ids,
                    log_probs=log_probs,
                    target_token_source=target_token_source,
                )
                key = logit_record_key(record)
                if key in completed_keys:
                    continue
                append_jsonl_record(args.output_file, record)
                completed_keys.add(key)
                output_records.append(record)
        return output_records

    prefix_records = read_jsonl(args.prefix_file)
    cases = shard_cases(select_logit_cases(prefix_records, args.logit_size, args.seed), args.shard_id, args.num_shards)
    for case in cases:
        _target_text, target_ids, target_token_source = target_token_ids_for_case(
            tokenizer,
            case,
            args.probe_tokens,
        )
        if not target_ids:
            continue
        contexts = {
            "c1_prefix_only_teacher": prefix_context_prompt(tokenizer, case, "prefix_only"),
            "c2_opsd_solution_oracle_teacher": prefix_context_prompt(tokenizer, case, "opsd_oracle"),
            "c3_intervention_oracle_teacher": prefix_context_prompt(tokenizer, case, "intervention"),
        }
        context_log_probs = {
            condition: compute_target_log_probs_hf(
                model=model,
                prompt_ids=_coerce_token_ids(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]),
                target_ids=target_ids,
                max_context_tokens=args.max_context_tokens,
            )
            for condition, prompt_text in contexts.items()
        }
        for contrast, teacher_condition in (
            ("opsd_solution_oracle_vs_prefix_only_teacher", "c2_opsd_solution_oracle_teacher"),
            ("intervention_oracle_vs_prefix_only_teacher", "c3_intervention_oracle_teacher"),
        ):
            key = f"kl_contrast:{case.get('case_id')}:{contrast}"
            if key in completed_keys:
                continue
            record = compare_contexts(
                case={**case, "target_condition": "c0_student_continue"},
                tokenizer=tokenizer,
                target_ids=target_ids,
                student_log_probs=context_log_probs["c1_prefix_only_teacher"],
                teacher_log_probs=context_log_probs[teacher_condition],
                contrast=contrast,
                top_k=args.top_k,
                top_kl_positions=args.top_kl_positions,
                first_window_tokens=args.first_window_tokens,
                target_token_source=target_token_source,
            )
            append_jsonl_record(args.output_file, record)
            completed_keys.add(key)
            output_records.append(record)
    return output_records


def rollout_context_prompt(
    tokenizer: Any,
    case: dict[str, Any],
    condition: str,
    skeletons: dict[int, dict[str, Any]],
) -> str:
    context_records = case.get("context_records")
    record = context_records.get(condition, case) if isinstance(context_records, dict) else case
    problem = str(record.get("problem") or case.get("problem") or "")
    solution = str(record.get("solution") or case.get("solution") or "")
    ground_truth = record.get("ground_truth", case.get("ground_truth"))

    if condition in {"student", "teacher_base"}:
        user_message = build_student_user_message(problem)
    elif condition == "teacher_reference":
        user_message = build_reference_user_message(problem, solution, ground_truth=ground_truth)
    elif condition == "teacher_skeleton":
        problem_id = int(record.get("problem_id", case.get("problem_id")))
        if problem_id not in skeletons:
            raise ValueError("--skeleton-file is required when probing teacher_skeleton contexts")
        user_message = build_semantic_skeleton_user_message(problem, skeletons[problem_id], ground_truth=ground_truth)
    else:
        raise ValueError(f"Unknown rollout context condition: {condition}")

    return render_chat_prompt(
        tokenizer,
        user_message,
        enable_thinking=(condition != "student"),
    )


def prefix_context_prompt(tokenizer: Any, case: dict[str, Any], prompt_kind: str) -> str:
    problem = str(case.get("problem") or "")
    if prompt_kind == "prefix_only":
        user_message = build_student_user_message(problem)
    elif prompt_kind == "opsd_oracle":
        user_message = build_opsd_oracle_user_message(problem, str(case.get("solution") or ""))
    elif prompt_kind == "intervention":
        user_message = build_intervention_user_message(problem, case.get("diagnostic") or build_heuristic_diagnostic(case.get("ground_truth")))
    else:
        raise ValueError(f"Unknown prefix prompt kind: {prompt_kind}")
    return render_prefill_prompt(
        tokenizer,
        user_message,
        assistant_prefix=str(case.get("student_prefix") or ""),
        enable_thinking=True,
    )


def compute_target_log_probs_hf(
    model: Any,
    prompt_ids: list[int],
    target_ids: list[int],
    max_context_tokens: int,
) -> Any:
    import torch

    max_prompt_tokens = max(1, max_context_tokens - len(target_ids))
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    input_ids = prompt_ids + list(target_ids)
    if len(input_ids) < 2:
        raise ValueError("Need at least one prompt token and one target token for logit probing")

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits[0]
    start = len(prompt_ids) - 1
    end = start + len(target_ids)
    target_logits = logits[start:end]
    if target_logits.shape[0] != len(target_ids):
        raise RuntimeError("HF model returned an unexpected number of target logit rows")
    return torch.log_softmax(target_logits.float(), dim=-1).cpu()


def top_token_ids(row: Any, top_k: int) -> list[int]:
    if top_k <= 0:
        return []
    if _is_tensor(row):
        values, indices = row.topk(min(top_k, row.shape[-1]))
        return [int(token_id) for token_id in indices.tolist()]
    return heapq.nlargest(top_k, row, key=row.get)


def build_top_token_rows(
    tokenizer: Any,
    log_probs: Any,
    top_token_ids_by_position: list[list[int]],
) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    for position, token_ids in enumerate(top_token_ids_by_position):
        position_rows: list[dict[str, Any]] = []
        for token_id in token_ids:
            token_logprob = _row_value(log_probs[position], int(token_id))
            position_rows.append(
                {
                    "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "prob": float(math.exp(token_logprob)),
                    "logprob": token_logprob,
                }
            )
        rows.append(position_rows)
    return rows


def tensor_compare_values(
    student_log_probs: Any,
    teacher_log_probs: Any,
    target_ids: list[int],
    top_k: int,
) -> tuple[list[float], list[float], list[float], list[float], list[list[int]], list[list[int]], list[float], list[float]]:
    import torch

    if student_log_probs.dim() != 2 or teacher_log_probs.dim() != 2:
        raise ValueError("tensor log-prob inputs must have shape [num_tokens, vocab_size]")
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student/base and teacher tensor log-prob shapes must match")
    if student_log_probs.shape[0] != len(target_ids):
        raise ValueError("log-prob row counts must match target token count")

    kl_values: list[float] = []
    teacher_entropy_values: list[float] = []
    student_entropy_values: list[float] = []
    delta_logp_values: list[float] = []
    teacher_top_ids_by_position: list[list[int]] = []
    student_top_ids_by_position: list[list[int]] = []

    with torch.no_grad():
        if student_log_probs.device != teacher_log_probs.device:
            student_log_probs = student_log_probs.to(teacher_log_probs.device)

        top_limit = min(top_k, int(teacher_log_probs.shape[-1])) if top_k > 0 else 0
        for start in range(0, len(target_ids), TENSOR_REDUCTION_CHUNK_TOKENS):
            end = min(start + TENSOR_REDUCTION_CHUNK_TOKENS, len(target_ids))
            student_lp = student_log_probs[start:end].float()
            teacher_lp = teacher_log_probs[start:end].float()
            target_tensor = torch.tensor(target_ids[start:end], dtype=torch.long, device=teacher_lp.device).unsqueeze(-1)

            teacher_probs = teacher_lp.exp()
            kl_tensor = (teacher_probs * (teacher_lp - student_lp)).sum(dim=-1)
            teacher_entropy_tensor = -(teacher_probs * teacher_lp).sum(dim=-1)
            del teacher_probs

            student_probs = student_lp.exp()
            student_entropy_tensor = -(student_probs * student_lp).sum(dim=-1)
            del student_probs

            delta_logp_tensor = (
                torch.gather(teacher_lp, dim=-1, index=target_tensor)
                - torch.gather(student_lp, dim=-1, index=target_tensor)
            ).squeeze(-1)

            kl_values.extend(float(value) for value in kl_tensor.cpu().tolist())
            teacher_entropy_values.extend(float(value) for value in teacher_entropy_tensor.cpu().tolist())
            student_entropy_values.extend(float(value) for value in student_entropy_tensor.cpu().tolist())
            delta_logp_values.extend(float(value) for value in delta_logp_tensor.cpu().tolist())

            if top_limit > 0:
                teacher_top_ids_by_position.extend(teacher_lp.topk(top_limit, dim=-1).indices.cpu().tolist())
                student_top_ids_by_position.extend(student_lp.topk(top_limit, dim=-1).indices.cpu().tolist())
            else:
                teacher_top_ids_by_position.extend([] for _ in range(start, end))
                student_top_ids_by_position.extend([] for _ in range(start, end))

    top1_matches = [
        float(teacher_top_ids[:1] == student_top_ids[:1])
        for teacher_top_ids, student_top_ids in zip(teacher_top_ids_by_position, student_top_ids_by_position)
    ]
    jaccards = []
    for teacher_top_ids, student_top_ids in zip(teacher_top_ids_by_position, student_top_ids_by_position):
        student_set = set(student_top_ids)
        teacher_set = set(teacher_top_ids)
        union = student_set | teacher_set
        jaccards.append(len(student_set & teacher_set) / len(union) if union else 0.0)

    return (
        kl_values,
        teacher_entropy_values,
        student_entropy_values,
        delta_logp_values,
        teacher_top_ids_by_position,
        student_top_ids_by_position,
        top1_matches,
        jaccards,
    )


def compare_contexts(
    case: dict[str, Any],
    tokenizer: Any,
    target_ids: list[int],
    student_log_probs: Any,
    teacher_log_probs: Any,
    contrast: str,
    top_k: int,
    top_kl_positions: int,
    first_window_tokens: int,
    target_token_source: str = TARGET_TOKEN_SOURCE_TEXT,
) -> dict[str, Any]:
    if len(student_log_probs) != len(target_ids) or len(teacher_log_probs) != len(target_ids):
        raise ValueError("log-prob row counts must match target token count")

    kl_values: list[float] = []
    teacher_entropy_values: list[float] = []
    student_entropy_values: list[float] = []
    delta_logp_values: list[float] = []
    top1_matches: list[float] = []
    jaccards: list[float] = []
    teacher_top_ids_by_position: list[list[int]] = []
    student_top_ids_by_position: list[list[int]] = []

    if _is_tensor(student_log_probs) and _is_tensor(teacher_log_probs):
        (
            kl_values,
            teacher_entropy_values,
            student_entropy_values,
            delta_logp_values,
            teacher_top_ids_by_position,
            student_top_ids_by_position,
            top1_matches,
            jaccards,
        ) = tensor_compare_values(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            target_ids=target_ids,
            top_k=top_k,
        )
    else:
        for pos, target_id in enumerate(target_ids):
            student_row = student_log_probs[pos]
            teacher_row = teacher_log_probs[pos]

            if _is_tensor(teacher_row):
                kl = _tensor_kl(teacher_row, student_row)
                teacher_entropy = _tensor_entropy(teacher_row)
                student_entropy = _tensor_entropy(student_row)
            else:
                if set(student_row) != set(teacher_row):
                    raise RuntimeError("Exact KL requires student/base and teacher log-prob rows over the same vocabulary.")
                kl = sum(
                    math.exp(teacher_lp) * (teacher_lp - student_row[token_id])
                    for token_id, teacher_lp in teacher_row.items()
                )
                teacher_entropy = -sum(math.exp(logprob) * logprob for logprob in teacher_row.values())
                student_entropy = -sum(math.exp(logprob) * logprob for logprob in student_row.values())

            if not _row_has_token(student_row, target_id) or not _row_has_token(teacher_row, target_id):
                raise RuntimeError(f"Target token id {target_id} is missing from log-prob rows.")
            delta_logp = _row_value(teacher_row, target_id) - _row_value(student_row, target_id)

            teacher_top_ids = top_token_ids(teacher_row, top_k)
            student_top_ids = top_token_ids(student_row, top_k)
            teacher_top_ids_by_position.append(teacher_top_ids)
            student_top_ids_by_position.append(student_top_ids)
            top1_matches.append(float(teacher_top_ids[:1] == student_top_ids[:1]))

            student_set = set(student_top_ids)
            teacher_set = set(teacher_top_ids)
            union = student_set | teacher_set
            jaccards.append(len(student_set & teacher_set) / len(union) if union else 0.0)

            kl_values.append(float(kl))
            teacher_entropy_values.append(float(teacher_entropy))
            student_entropy_values.append(float(student_entropy))
            delta_logp_values.append(float(delta_logp))

    token_texts = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in target_ids]
    style_mask = [is_style_token(text) for text in token_texts]
    math_mask = [is_math_token(text) for text in token_texts]
    total_kl = float(sum(kl_values))
    first_window = min(first_window_tokens, len(target_ids))
    delta_entropy_values = [
        teacher_entropy - student_entropy
        for teacher_entropy, student_entropy in zip(teacher_entropy_values, student_entropy_values)
    ]
    teacher_top_token_rows = build_top_token_rows(tokenizer, teacher_log_probs, teacher_top_ids_by_position)
    base_top_token_rows = build_top_token_rows(tokenizer, student_log_probs, student_top_ids_by_position)

    return {
        "record_type": "kl_contrast",
        "logprob_backend": LOGPROB_BACKEND,
        "target_token_source": target_token_source,
        "case_id": case.get("case_id"),
        "problem_id": case.get("problem_id"),
        "contrast": contrast,
        "target_condition": case.get("target_condition"),
        "target_sample_index": case.get("target_sample_index", case.get("sample_index")),
        "prefix_token_cutoff": case.get("prefix_token_cutoff"),
        "num_tokens": len(target_ids),
        "top_k": top_k,
        "mean_kl": sum(kl_values) / len(kl_values) if kl_values else 0.0,
        "sum_kl": total_kl,
        "top1_agreement": sum(top1_matches) / len(top1_matches) if top1_matches else 0.0,
        "topk_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "mean_delta_logp_target": sum(delta_logp_values) / len(delta_logp_values) if delta_logp_values else 0.0,
        "mean_teacher_entropy": (
            sum(teacher_entropy_values) / len(teacher_entropy_values) if teacher_entropy_values else 0.0
        ),
        "mean_student_entropy": (
            sum(student_entropy_values) / len(student_entropy_values) if student_entropy_values else 0.0
        ),
        "mean_delta_entropy": sum(delta_entropy_values) / len(delta_entropy_values) if delta_entropy_values else 0.0,
        "style_kl_share": _masked_share(kl_values, style_mask, total_kl),
        "math_kl_share": _masked_share(kl_values, math_mask, total_kl),
        "first_window_kl_share": (
            sum(kl_values[:first_window]) / total_kl if total_kl > 0 and first_window else 0.0
        ),
        "token_texts": token_texts,
        "kl_per_token": kl_values,
        "delta_logp_target_per_token": delta_logp_values,
        "teacher_entropy_per_token": teacher_entropy_values,
        "student_entropy_per_token": student_entropy_values,
        "delta_entropy_per_token": delta_entropy_values,
        "top_kl_positions": build_top_kl_positions(
            token_texts=token_texts,
            kl_values=kl_values,
            delta_logp_values=delta_logp_values,
            teacher_entropy_values=teacher_entropy_values,
            student_entropy_values=student_entropy_values,
            teacher_top_token_rows=teacher_top_token_rows,
            base_top_token_rows=base_top_token_rows,
            top_n=top_kl_positions,
        ),
    }


def build_rollout_entropy_record(
    case: dict[str, Any],
    tokenizer: Any,
    target_ids: list[int],
    log_probs: Any,
    target_token_source: str = TARGET_TOKEN_SOURCE_TEXT,
) -> dict[str, Any]:
    if _is_tensor(log_probs):
        entropy_values = []
        for start in range(0, len(target_ids), TENSOR_REDUCTION_CHUNK_TOKENS):
            end = min(start + TENSOR_REDUCTION_CHUNK_TOKENS, len(target_ids))
            log_probs_float = log_probs[start:end].float()
            entropy_values.extend(
                float(value)
                for value in (-(log_probs_float.exp() * log_probs_float).sum(dim=-1)).cpu().tolist()
            )
    else:
        entropy_values = [_row_entropy(log_probs[position]) for position in range(len(target_ids))]
    return {
        "record_type": "rollout_entropy",
        "logprob_backend": LOGPROB_BACKEND,
        "target_token_source": target_token_source,
        "case_id": case.get("case_id"),
        "problem_id": case.get("problem_id"),
        "condition": case.get("target_condition", case.get("condition")),
        "target_sample_index": case.get("target_sample_index", case.get("sample_index")),
        "num_tokens": len(target_ids),
        "mean_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0.0,
        "entropy_per_token": entropy_values,
        "token_texts": [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in target_ids],
    }


def build_top_kl_positions(
    token_texts: list[str],
    kl_values: list[float],
    delta_logp_values: list[float],
    teacher_entropy_values: list[float],
    student_entropy_values: list[float],
    top_n: int,
    teacher_top_token_rows: list[list[dict[str, Any]]] | None = None,
    base_top_token_rows: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for position, (token_text, kl, delta_logp, teacher_entropy, student_entropy) in enumerate(
        zip(token_texts, kl_values, delta_logp_values, teacher_entropy_values, student_entropy_values)
    ):
        rows.append(
            {
                "position": position,
                "token_text": token_text,
                "kl": float(kl),
                "delta_logp_target": float(delta_logp),
                "teacher_entropy": float(teacher_entropy),
                "student_entropy": float(student_entropy),
                "delta_entropy": float(teacher_entropy) - float(student_entropy),
                "teacher_top_tokens": (
                    teacher_top_token_rows[position]
                    if teacher_top_token_rows is not None and position < len(teacher_top_token_rows)
                    else []
                ),
                "base_top_tokens": (
                    base_top_token_rows[position]
                    if base_top_token_rows is not None and position < len(base_top_token_rows)
                    else []
                ),
            }
        )
    rows.sort(key=lambda item: item["kl"], reverse=True)
    return rows[: max(0, top_n)]


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "dim") and hasattr(value, "shape")


def _row_value(row: Any, token_id: int) -> float:
    if _is_tensor(row):
        return float(row[int(token_id)].item())
    return float(row[int(token_id)])


def _row_has_token(row: Any, token_id: int) -> bool:
    if _is_tensor(row):
        return 0 <= int(token_id) < int(row.shape[-1])
    return int(token_id) in row


def _tensor_kl(teacher_row: Any, student_row: Any) -> float:
    return float((teacher_row.exp() * (teacher_row - student_row)).sum().item())


def _tensor_entropy(row: Any) -> float:
    return float((-(row.exp() * row)).sum().item())


def _row_entropy(row: Any) -> float:
    if _is_tensor(row):
        return _tensor_entropy(row)
    return float(-sum(math.exp(logprob) * logprob for logprob in row.values()))


def _masked_share(values: Any, mask: Any, total: float) -> float:
    if total <= 0 or not any(mask):
        return 0.0
    return sum(value for value, keep in zip(values, mask) if keep) / total


if __name__ == "__main__":
    main()
