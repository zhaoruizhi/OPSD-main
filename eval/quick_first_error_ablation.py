#!/usr/bin/env python
"""First-error ablation probe for OPSD quick runs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .quick_logit_probe import compare_contexts, compute_target_log_probs_hf
    from .quick_opsd_common import (
        build_first_error_user_message,
        build_reference_user_message,
        build_student_user_message,
        continuation_metrics,
        first_error_text_slices,
        first_error_token_ranges,
        read_first_error_file,
        read_jsonl,
        render_chat_prompt,
        render_prefill_prompt,
        shard_items,
        summarize_generation_records,
        summarize_logit_records,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - used when run as python eval/script.py
    from quick_logit_probe import compare_contexts, compute_target_log_probs_hf
    from quick_opsd_common import (
        build_first_error_user_message,
        build_reference_user_message,
        build_student_user_message,
        continuation_metrics,
        first_error_text_slices,
        first_error_token_ranges,
        read_first_error_file,
        read_jsonl,
        render_chat_prompt,
        render_prefill_prompt,
        shard_items,
        summarize_generation_records,
        summarize_logit_records,
        write_json,
        write_jsonl,
    )


TARGET_TOKEN_SOURCE_TEXT = "target_tail_text"
TARGET_TOKEN_SOURCE_TOKEN_IDS = "completion_token_ids"


@dataclass(frozen=True)
class FirstErrorConditionSpec:
    name: str
    enable_thinking: bool
    prompt_kind: str


def build_first_error_condition_specs() -> list[FirstErrorConditionSpec]:
    return [
        FirstErrorConditionSpec("teacher_base_w_text", enable_thinking=True, prompt_kind="reference_text"),
        FirstErrorConditionSpec(
            "teacher_base_w_first_error",
            enable_thinking=True,
            prompt_kind="first_error_json",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-error OPSD ablations.")
    parser.add_argument("--mode", choices=["continuation", "kl"], default="continuation")
    parser.add_argument("--model", type=str, default="/data0/shared/Qwen3-1.7B")
    parser.add_argument("--student-rollout-file", type=str)
    parser.add_argument("--first-error-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--summary-file", type=str, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summary-kind", choices=["generation", "kl"], default="generation")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--case-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--val-n", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--hf-device-map", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--top-kl-positions", type=int, default=20)
    parser.add_argument("--first-window-tokens", type=int, default=32)
    parser.add_argument("--neighborhood-before-tokens", type=int, default=32)
    parser.add_argument("--neighborhood-after-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.summarize_only:
        if not args.input_file:
            parser.error("--input-file is required with --summarize-only")
        return args
    if not args.student_rollout_file or not args.first_error_file:
        parser.error("--student-rollout-file and --first-error-file are required")
    if not args.output_file:
        parser.error("--output-file is required unless --summarize-only is set")
    return args


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        records = read_jsonl(args.input_file)
        if args.summary_kind == "kl":
            write_json(args.summary_file, summarize_first_error_kl_records(records))
        else:
            write_json(args.summary_file, summarize_generation_records(records))
        return

    if args.mode == "continuation":
        records = run_continuations(args)
        write_jsonl(args.output_file, records)
        write_json(args.summary_file, summarize_generation_records(records))
    else:
        records = run_segmented_kl(args)
        write_jsonl(args.output_file, records)
        write_json(args.summary_file, summarize_first_error_kl_records(records))


def _int_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(token_id) for token_id in value]
    except TypeError:
        return []


def _encode_prompt_token_ids(tokenizer: Any, prompt_text: str) -> list[int]:
    encoded = tokenizer(prompt_text, add_special_tokens=False)
    return _int_token_ids(encoded.get("input_ids"))


def select_first_error_cases(
    tokenizer: Any,
    rollout_records: list[dict[str, Any]],
    diagnostics: dict[int, dict[str, Any]],
    case_size: int,
    seed: int,
    neighborhood_before: int,
    neighborhood_after: int,
) -> list[dict[str, Any]]:
    first_by_problem: dict[int, dict[str, Any]] = {}
    for record in sorted(
        rollout_records,
        key=lambda item: (int(item.get("problem_id", -1)), int(item.get("sample_index", 0))),
    ):
        if record.get("condition") != "student":
            continue
        try:
            sample_index = int(record.get("sample_index", 0))
            problem_id = int(record["problem_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if sample_index != 0 or problem_id not in diagnostics:
            continue
        first_by_problem.setdefault(problem_id, record)

    cases: list[dict[str, Any]] = []
    for problem_id, record in sorted(first_by_problem.items()):
        diagnostic = diagnostics[problem_id]
        if diagnostic.get("first_error_sentence") is None:
            continue
        full_generation = str(record.get("full_generation") or "")
        if not full_generation:
            continue
        text_slices = first_error_text_slices(full_generation, diagnostic)
        token_ranges = first_error_token_ranges(
            tokenizer,
            full_generation,
            diagnostic,
            neighborhood_before=neighborhood_before,
            neighborhood_after=neighborhood_after,
        )
        completion_token_ids = _int_token_ids(record.get("completion_token_ids"))
        target_ids = completion_token_ids or token_ranges["target_token_ids"]
        if not target_ids:
            continue
        case = {
            **record,
            "case_id": f"{problem_id}:0:first_error",
            "problem_id": problem_id,
            "source_student_sample_index": 0,
            "diagnostic": diagnostic,
            "student_prefix": text_slices["student_prefix"],
            "target_tail_text": text_slices["target_tail_text"],
            "prefix_char_end": text_slices["prefix_char_end"],
            "first_error_char_range": text_slices["first_error_char_range"],
            "prefix_valid_until": diagnostic["prefix_valid_until"],
            "first_error_sentence": diagnostic["first_error_sentence"],
            "target_token_ids": target_ids,
            "target_token_source": (
                TARGET_TOKEN_SOURCE_TOKEN_IDS if completion_token_ids else TARGET_TOKEN_SOURCE_TEXT
            ),
            **{key: value for key, value in token_ranges.items() if key != "target_token_ids"},
        }
        cases.append(case)

    if case_size > 0 and len(cases) > case_size:
        rng = random.Random(seed)
        cases = rng.sample(cases, case_size)
    return sorted(cases, key=lambda item: int(item.get("problem_id", -1)))


def user_message_for_condition(spec: FirstErrorConditionSpec, case: dict[str, Any]) -> str:
    problem = str(case.get("problem") or "")
    ground_truth = case.get("ground_truth")
    if spec.prompt_kind == "reference_text":
        return build_reference_user_message(problem, str(case.get("solution") or ""), ground_truth=ground_truth)
    if spec.prompt_kind == "first_error_json":
        return build_first_error_user_message(problem, case["diagnostic"], ground_truth=ground_truth)
    raise ValueError(f"Unknown first-error prompt kind: {spec.prompt_kind}")


def run_continuations(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = select_first_error_cases(
        tokenizer=tokenizer,
        rollout_records=read_jsonl(args.student_rollout_file),
        diagnostics=read_first_error_file(args.first_error_file),
        case_size=args.case_size,
        seed=args.seed,
        neighborhood_before=args.neighborhood_before_tokens,
        neighborhood_after=args.neighborhood_after_tokens,
    )
    shard_cases_list = shard_items(cases, args.shard_id, args.num_shards)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        distributed_executor_backend="mp",
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=args.val_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )

    output_records: list[dict[str, Any]] = []
    for spec in build_first_error_condition_specs():
        prompts = [
            render_prefill_prompt(
                tokenizer,
                user_message_for_condition(spec, case),
                assistant_prefix=case["student_prefix"],
                enable_thinking=spec.enable_thinking,
            )
            for case in shard_cases_list
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for case, prompt_text, output in zip(shard_cases_list, prompts, outputs):
            prompt_token_ids = _int_token_ids(getattr(output, "prompt_token_ids", None))
            if not prompt_token_ids:
                prompt_token_ids = _encode_prompt_token_ids(tokenizer, prompt_text)
            for sample_index, completion in enumerate(output.outputs):
                continuation = completion.text
                completion_token_ids = _int_token_ids(getattr(completion, "token_ids", None))
                metrics = continuation_metrics(
                    prefix=case["student_prefix"],
                    continuation=continuation,
                    ground_truth=case.get("ground_truth"),
                    reference_solution=case.get("solution"),
                )
                output_records.append(
                    {
                        "case_id": case["case_id"],
                        "problem_id": case.get("problem_id"),
                        "problem": case.get("problem"),
                        "solution": case.get("solution"),
                        "ground_truth": case.get("ground_truth"),
                        "source_student_sample_index": case.get("source_student_sample_index", 0),
                        "condition": spec.name,
                        "enable_thinking": spec.enable_thinking,
                        "sample_index": sample_index,
                        "diagnostic": case["diagnostic"],
                        "prefix_valid_until": case["prefix_valid_until"],
                        "first_error_sentence": case["first_error_sentence"],
                        "student_prefix": case["student_prefix"],
                        "target_tail_text": case["target_tail_text"],
                        "prefix_token_cutoff": case["prefix_token_cutoff"],
                        "valid_prefix_range": case["valid_prefix_range"],
                        "first_error_range": case["first_error_range"],
                        "first_error_neighborhood_range": case["first_error_neighborhood_range"],
                        "continuation": continuation,
                        "full_generation": f"{case['student_prefix']}{continuation}",
                        "prompt_token_ids": prompt_token_ids,
                        "completion_token_ids": completion_token_ids,
                        "completion_tokens": len(completion_token_ids),
                        "finish_reason": getattr(completion, "finish_reason", None),
                        **metrics,
                    }
                )
    return output_records


def run_segmented_kl(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.hf_device_map == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--hf-device-map cuda requires CUDA, but torch.cuda.is_available() is False")
        model_kwargs["device_map"] = {"": "cuda:0"}
    elif args.hf_device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    cases = select_first_error_cases(
        tokenizer=tokenizer,
        rollout_records=read_jsonl(args.student_rollout_file),
        diagnostics=read_first_error_file(args.first_error_file),
        case_size=args.case_size,
        seed=args.seed,
        neighborhood_before=args.neighborhood_before_tokens,
        neighborhood_after=args.neighborhood_after_tokens,
    )
    shard_cases_list = shard_items(cases, args.shard_id, args.num_shards)

    output_records: list[dict[str, Any]] = []
    for case in shard_cases_list:
        target_ids = list(case["target_token_ids"])
        if not target_ids:
            continue
        base_prompt_ids = _int_token_ids(case.get("prompt_token_ids"))
        if not base_prompt_ids:
            base_prompt = render_chat_prompt(
                tokenizer,
                build_student_user_message(str(case.get("problem") or "")),
                enable_thinking=False,
            )
            base_prompt_ids = _encode_prompt_token_ids(tokenizer, base_prompt)
        base_log_probs = compute_target_log_probs_hf(
            model=model,
            prompt_ids=base_prompt_ids,
            target_ids=target_ids,
            max_context_tokens=args.max_model_len,
        )

        for spec in build_first_error_condition_specs():
            teacher_prompt = render_chat_prompt(
                tokenizer,
                user_message_for_condition(spec, case),
                enable_thinking=spec.enable_thinking,
            )
            teacher_prompt_ids = _encode_prompt_token_ids(tokenizer, teacher_prompt)
            teacher_log_probs = compute_target_log_probs_hf(
                model=model,
                prompt_ids=teacher_prompt_ids,
                target_ids=target_ids,
                max_context_tokens=args.max_model_len,
            )
            record = compare_contexts(
                case={**case, "target_condition": "student"},
                tokenizer=tokenizer,
                target_ids=target_ids,
                student_log_probs=base_log_probs,
                teacher_log_probs=teacher_log_probs,
                contrast=f"{spec.name}_vs_student_base",
                top_k=args.top_k,
                top_kl_positions=args.top_kl_positions,
                first_window_tokens=args.first_window_tokens,
                target_token_source=case["target_token_source"],
            )
            record.update(
                {
                    "prefix_valid_until": case["prefix_valid_until"],
                    "first_error_sentence": case["first_error_sentence"],
                    "valid_prefix_range": case["valid_prefix_range"],
                    "first_error_range": case["first_error_range"],
                    "first_error_neighborhood_range": case["first_error_neighborhood_range"],
                    "segment_kl": segment_kl_summary(record, case),
                }
            )
            output_records.append(record)
    return output_records


def segment_kl_summary(record: dict[str, Any], case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "valid_prefix": summarize_kl_slice(record, case["valid_prefix_range"]),
        "first_error_neighborhood": summarize_kl_slice(record, case["first_error_neighborhood_range"]),
    }


def summarize_kl_slice(record: dict[str, Any], token_range: list[int]) -> dict[str, Any]:
    start, end = [int(value) for value in token_range]
    start = max(0, start)
    end = min(max(start, end), len(record.get("kl_per_token", [])))
    metrics = {
        "kl_per_token": record.get("kl_per_token", []),
        "delta_logp_target_per_token": record.get("delta_logp_target_per_token", []),
        "teacher_entropy_per_token": record.get("teacher_entropy_per_token", []),
        "student_entropy_per_token": record.get("student_entropy_per_token", []),
        "delta_entropy_per_token": record.get("delta_entropy_per_token", []),
    }
    values = {name: list(items[start:end]) for name, items in metrics.items()}
    return {
        "range": [start, end],
        "num_tokens": end - start,
        "mean_kl": _mean(values["kl_per_token"]),
        "sum_kl": float(sum(values["kl_per_token"])),
        "mean_delta_logp_target": _mean(values["delta_logp_target_per_token"]),
        "mean_teacher_entropy": _mean(values["teacher_entropy_per_token"]),
        "mean_student_entropy": _mean(values["student_entropy_per_token"]),
        "mean_delta_entropy": _mean(values["delta_entropy_per_token"]),
    }


def summarize_first_error_kl_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_logit_records(records)
    by_contrast: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_contrast.setdefault(str(record.get("contrast", "unknown")), []).append(record)

    summary["segment_kl"] = {}
    for contrast, items in sorted(by_contrast.items()):
        summary["segment_kl"][contrast] = {}
        for segment_name in ("valid_prefix", "first_error_neighborhood"):
            segment_items = [
                item.get("segment_kl", {}).get(segment_name, {})
                for item in items
                if isinstance(item.get("segment_kl", {}).get(segment_name), dict)
            ]
            summary["segment_kl"][contrast][segment_name] = {
                "num_cases": len(segment_items),
                "mean_num_tokens": _mean(item.get("num_tokens") for item in segment_items),
                "mean_kl": _mean(item.get("mean_kl") for item in segment_items),
                "mean_sum_kl": _mean(item.get("sum_kl") for item in segment_items),
                "mean_delta_logp_target": _mean(
                    item.get("mean_delta_logp_target") for item in segment_items
                ),
                "mean_teacher_entropy": _mean(item.get("mean_teacher_entropy") for item in segment_items),
                "mean_student_entropy": _mean(item.get("mean_student_entropy") for item in segment_items),
                "mean_delta_entropy": _mean(item.get("mean_delta_entropy") for item in segment_items),
            }
    return summary


def _mean(values: Any) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


if __name__ == "__main__":
    main()
