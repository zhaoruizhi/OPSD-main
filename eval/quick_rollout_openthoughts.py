#!/usr/bin/env python
"""Standalone rollout probe for OPSD quick runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .quick_opsd_common import (
        build_reference_user_message,
        build_semantic_skeleton_user_message,
        build_student_user_message,
        choose_stratified_indices,
        continuation_metrics,
        extract_boxed_answer,
        get_ground_truth_answer,
        get_problem_text,
        get_solution_text,
        read_jsonl,
        read_sample_indices_file,
        read_skeleton_file,
        render_chat_prompt,
        shard_items,
        summarize_generation_records,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - used when run as python eval/script.py
    from quick_opsd_common import (
        build_reference_user_message,
        build_semantic_skeleton_user_message,
        build_student_user_message,
        choose_stratified_indices,
        continuation_metrics,
        extract_boxed_answer,
        get_ground_truth_answer,
        get_problem_text,
        get_solution_text,
        read_jsonl,
        read_sample_indices_file,
        read_skeleton_file,
        render_chat_prompt,
        shard_items,
        summarize_generation_records,
        write_json,
        write_jsonl,
    )


@dataclass(frozen=True)
class RolloutConditionSpec:
    name: str
    enable_thinking: bool
    prompt_kind: str


def build_rollout_condition_specs() -> list[RolloutConditionSpec]:
    return [
        RolloutConditionSpec("student", enable_thinking=False, prompt_kind="student"),
        RolloutConditionSpec("teacher_base", enable_thinking=True, prompt_kind="base"),
        RolloutConditionSpec("teacher_reference", enable_thinking=True, prompt_kind="reference"),
        RolloutConditionSpec("teacher_skeleton", enable_thinking=True, prompt_kind="skeleton"),
    ]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OPSD quick standalone rollouts on OpenThoughts data.")
    parser.add_argument("--model", type=str, default="/data0/shared/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--sample-indices-file", type=str)
    parser.add_argument("--skeleton-file", type=str)
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
    parser.add_argument("--output-file", type=str, required=False)
    parser.add_argument("--summary-file", type=str, required=True)
    parser.add_argument(
        "--condition",
        action="append",
        choices=[spec.name for spec in build_rollout_condition_specs()],
        help="Condition to run. Repeatable. Defaults to all conditions.",
    )
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

    if not args.output_file:
        raise ValueError("--output-file is required unless --summarize-only is set")

    records = run_rollouts(args)
    write_jsonl(args.output_file, records)
    write_json(args.summary_file, summarize_generation_records(records))


def run_rollouts(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dataset = load_dataset(args.dataset, split=args.split)
    all_rows = [dict(row) for row in dataset]
    if args.sample_indices_file:
        selected_indices = read_sample_indices_file(args.sample_indices_file)
    else:
        selected_indices = choose_stratified_indices(all_rows, args.sample_size, args.seed)
    shard_indices = shard_items(selected_indices, args.shard_id, args.num_shards)
    selected_examples = [(idx, all_rows[idx]) for idx in shard_indices]
    skeletons = read_skeleton_file(args.skeleton_file) if args.skeleton_file else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    requested = set(args.condition or [spec.name for spec in build_rollout_condition_specs()])
    if "teacher_skeleton" in requested and not skeletons:
        raise ValueError("--skeleton-file is required when running teacher_skeleton")
    output_records: list[dict[str, Any]] = []
    for spec in build_rollout_condition_specs():
        if spec.name not in requested:
            continue

        prompts: list[str] = []
        metadata: list[dict[str, Any]] = []
        for original_index, example in selected_examples:
            problem = get_problem_text(example)
            solution = get_solution_text(example)
            ground_truth = get_ground_truth_answer(example)
            user_message = user_message_for_rollout(
                spec=spec,
                problem=problem,
                solution=solution,
                skeleton=skeletons.get(int(original_index)),
                answer=ground_truth,
                problem_id=int(original_index),
            )
            prompt_text = render_chat_prompt(tokenizer, user_message, enable_thinking=spec.enable_thinking)
            prompts.append(prompt_text)
            metadata.append(
                {
                    "_prompt_text": prompt_text,
                    "problem_id": original_index,
                    "problem": problem,
                    "solution": solution,
                    "ground_truth": ground_truth,
                    "source": example.get("source"),
                    "generated_token_count": example.get("generated_token_count"),
                }
            )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        for meta, output in zip(metadata, outputs):
            prompt_token_ids = _int_token_ids(getattr(output, "prompt_token_ids", None))
            if not prompt_token_ids:
                prompt_token_ids = _encode_prompt_token_ids(tokenizer, str(meta.get("_prompt_text") or ""))
            record_meta = {key: value for key, value in meta.items() if not key.startswith("_")}
            for sample_index, completion in enumerate(output.outputs):
                text = completion.text
                completion_token_ids = _int_token_ids(getattr(completion, "token_ids", None))
                predicted = extract_boxed_answer(text)
                metrics = continuation_metrics(
                    prefix="",
                    continuation=text,
                    ground_truth=record_meta["ground_truth"],
                    reference_solution=record_meta["solution"],
                )
                output_records.append(
                    {
                        **record_meta,
                        "condition": spec.name,
                        "enable_thinking": spec.enable_thinking,
                        "sample_index": sample_index,
                        "predicted_answer": predicted,
                        "full_generation": text,
                        "prompt_token_ids": prompt_token_ids,
                        "completion_token_ids": completion_token_ids,
                        "formatted": metrics["formatted"],
                        "correct": metrics["correct"],
                        "restart": metrics["restart"],
                        "prefix_preserved": metrics["prefix_preserved"],
                        "notation_consistency": metrics["notation_consistency"],
                        "locality_score": metrics["locality_score"],
                        "reference_copy_rate": metrics["reference_copy_rate"],
                        "completion_tokens": len(completion_token_ids),
                        "finish_reason": getattr(completion, "finish_reason", None),
                    }
                )

    return output_records


def user_message_for_rollout(
    spec: RolloutConditionSpec,
    problem: str,
    solution: str,
    skeleton: dict[str, Any] | None,
    answer: str | None,
    problem_id: int,
) -> str:
    if spec.prompt_kind == "student":
        return build_student_user_message(problem)
    if spec.prompt_kind == "base":
        return build_student_user_message(problem)
    if spec.prompt_kind == "reference":
        return build_reference_user_message(problem, solution, answer=answer)
    if spec.prompt_kind == "skeleton":
        if skeleton is None:
            raise ValueError(f"missing semantic skeleton for problem_id={problem_id}")
        return build_semantic_skeleton_user_message(problem, skeleton, answer=answer)
    raise ValueError(f"Unknown prompt kind: {spec.prompt_kind}")


if __name__ == "__main__":
    main()
