#!/usr/bin/env python
"""Prefix-conditioned continuation probe for OPSD quick runs."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any

try:
    from .quick_opsd_common import (
        build_heuristic_diagnostic,
        build_intervention_user_message,
        build_opsd_oracle_user_message,
        build_student_user_message,
        continuation_metrics,
        read_jsonl,
        render_prefill_prompt,
        shard_items,
        split_prefix_by_token_ratio,
        summarize_generation_records,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover
    from quick_opsd_common import (
        build_heuristic_diagnostic,
        build_intervention_user_message,
        build_opsd_oracle_user_message,
        build_student_user_message,
        continuation_metrics,
        read_jsonl,
        render_prefill_prompt,
        shard_items,
        split_prefix_by_token_ratio,
        summarize_generation_records,
        write_json,
        write_jsonl,
    )


@dataclass(frozen=True)
class PrefixConditionSpec:
    name: str
    enable_thinking: bool
    prompt_kind: str


def build_prefix_condition_specs() -> list[PrefixConditionSpec]:
    return [
        PrefixConditionSpec("c0_student_continue", enable_thinking=False, prompt_kind="student"),
        PrefixConditionSpec("c1_prefix_only_teacher", enable_thinking=True, prompt_kind="student"),
        PrefixConditionSpec("c2_opsd_solution_oracle_teacher", enable_thinking=True, prompt_kind="opsd_oracle"),
        PrefixConditionSpec("c3_intervention_oracle_teacher", enable_thinking=True, prompt_kind="intervention"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPSD quick prefix intervention continuations.")
    parser.add_argument("--model", type=str, default="/data0/shared/Qwen3-1.7B")
    parser.add_argument("--student-rollout-file", type=str, required=False)
    parser.add_argument("--prefix-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prefix-ratio", type=float, default=0.5)
    parser.add_argument("--val-n", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--summary-file", type=str, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--input-file", type=str, help="JSONL file to summarize when --summarize-only is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        if not args.input_file:
            raise ValueError("--input-file is required with --summarize-only")
        records = read_jsonl(args.input_file)
        write_json(args.summary_file, summarize_generation_records(records))
        return

    if not args.student_rollout_file or not args.output_file:
        raise ValueError("--student-rollout-file and --output-file are required unless --summarize-only is set")

    records = run_prefix_probe(args)
    write_jsonl(args.output_file, records)
    write_json(args.summary_file, summarize_generation_records(records))


def select_student_cases(records: list[dict[str, Any]], prefix_size: int, seed: int) -> list[dict[str, Any]]:
    first_by_problem: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (str(item.get("problem_id")), int(item.get("sample_index", 0)))):
        if record.get("condition") != "student_full":
            continue
        problem_id = str(record.get("problem_id"))
        first_by_problem.setdefault(problem_id, record)

    cases = list(first_by_problem.values())
    if prefix_size > 0 and len(cases) > prefix_size:
        rng = random.Random(seed)
        cases = rng.sample(cases, prefix_size)
    return sorted(cases, key=lambda item: str(item.get("problem_id")))


def user_message_for_condition(spec: PrefixConditionSpec, case: dict[str, Any]) -> str:
    problem = str(case["problem"])
    if spec.prompt_kind == "student":
        return build_student_user_message(problem)
    if spec.prompt_kind == "opsd_oracle":
        return build_opsd_oracle_user_message(problem, str(case.get("solution") or ""))
    if spec.prompt_kind == "intervention":
        diagnostic = build_heuristic_diagnostic(case.get("ground_truth"))
        return build_intervention_user_message(problem, diagnostic)
    raise ValueError(f"Unknown prompt kind: {spec.prompt_kind}")


def run_prefix_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rollout_records = read_jsonl(args.student_rollout_file)
    selected_cases = select_student_cases(rollout_records, args.prefix_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prepared_cases = []
    for case in selected_cases:
        prefix, target_tail, cutoff = split_prefix_by_token_ratio(
            tokenizer,
            str(case.get("full_generation") or ""),
            ratio=args.prefix_ratio,
        )
        if not prefix:
            continue
        prepared = {
            **case,
            "case_id": f"{case.get('problem_id')}:{case.get('sample_index', 0)}",
            "student_prefix": prefix,
            "target_tail_text": target_tail,
            "prefix_token_cutoff": cutoff,
            "prefix_ratio": args.prefix_ratio,
            "diagnostic": build_heuristic_diagnostic(case.get("ground_truth")),
        }
        prepared_cases.append(prepared)

    shard_cases = shard_items(prepared_cases, args.shard_id, args.num_shards)

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
    for spec in build_prefix_condition_specs():
        prompts = [
            render_prefill_prompt(
                tokenizer,
                user_message_for_condition(spec, case),
                assistant_prefix=case["student_prefix"],
                enable_thinking=spec.enable_thinking,
            )
            for case in shard_cases
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for case, output in zip(shard_cases, outputs):
            for sample_index, completion in enumerate(output.outputs):
                continuation = completion.text
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
                        "source_student_sample_index": case.get("sample_index", 0),
                        "condition": spec.name,
                        "enable_thinking": spec.enable_thinking,
                        "sample_index": sample_index,
                        "student_prefix": case["student_prefix"],
                        "target_tail_text": case["target_tail_text"],
                        "prefix_token_cutoff": case["prefix_token_cutoff"],
                        "diagnostic": case["diagnostic"],
                        "continuation": continuation,
                        "full_generation": f"{case['student_prefix']}{continuation}",
                        "predicted_answer": metrics["predicted_answer"],
                        "formatted": metrics["formatted"],
                        "correct": metrics["correct"],
                        "restart": metrics["restart"],
                        "prefix_preserved": metrics["prefix_preserved"],
                        "notation_consistency": metrics["notation_consistency"],
                        "locality_score": metrics["locality_score"],
                        "reference_copy_rate": metrics["reference_copy_rate"],
                        "completion_tokens": len(getattr(completion, "token_ids", []) or []),
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                )

    return output_records


if __name__ == "__main__":
    main()
