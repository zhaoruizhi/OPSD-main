#!/usr/bin/env python
"""Generate style-neutral semantic skeletons from reference solutions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

try:
    from .quick_opsd_common import (
        get_ground_truth_answer,
        get_solution_text,
        normalize_semantic_skeleton,
        read_sample_indices_file,
        write_jsonl,
    )
except ImportError:  # pragma: no cover
    from quick_opsd_common import (
        get_ground_truth_answer,
        get_solution_text,
        normalize_semantic_skeleton,
        read_sample_indices_file,
        write_jsonl,
    )


SYSTEM_PROMPT = """You are a mathematical semantic-skeleton compiler.
Your task is to convert a reference solution and its final answer into a
style-neutral, structured representation of the mathematical content.

The input may NOT contain the original problem statement. Therefore:

1. Describe mathematical objectives rather than copying sentences from the
   reference solution.
2. If information is missing, use an empty list, null, or mark the item as
   uncertain. Never fill missing information with plausible guesses.
3. Return exactly one valid JSON object. Do not use Markdown, code fences,
   comments, or text outside the JSON object.

Produce the following schema:

{
  "final_answer": "...",
  "key_objects": [
    {
        "name": "..., ...",
        "constraints": ["...", "..."]
    }
  ],
  "subgoals": [
        "...",
        "...",
        "..."
  ],
  "critical_intermediates": [
    "...",
    "..."
  ],
  "theorem_tags": [
    "...",
    "..."
  ],
  "checks": [
    "...",
    "..."
  ]
}

Field-specific rules:

A. final_answer
- Copy directly from ANSWER.

B. key_objects
- Include only objects central to the reasoning.
- Do not list every symbol appearing in the solution.
- Prefer at most 8 objects.
- Preserve original notation whenever possible.

C. subgoals
- Express each subgoal as a concise mathematical objective.
- Do not copy procedural prose such as "Next, we calculate..." or
  "We can easily see...".
- Prefer 2 to 8 subgoals.
- A subgoal should state what must be established, not the exact sentence
  the teacher should generate.

D. critical_intermediates
- Include only relations whose correctness materially affects the argument.
- Prefer at most 5 intermediate results.

E. theorem_tags
- Use short canonical names such as:
  "Cauchy-Schwarz inequality", "induction",
  "case split", "contradiction" or "boundary analysis".
- Include a tag only when it is actually used or strongly implied.
- Prefer at most 5 tags.

F. checks
- Include only concrete failure modes relevant to this solution.
- Examples include dividing by a potentially zero expression, losing a case,
  reversing a non-equivalent implication, ignoring a domain condition,
  introducing extraneous roots, or applying a theorem outside its conditions.
- Prefer at most 5 items."""


def build_skeleton_compiler_prompt(answer: str | None, reference_solution: str) -> str:
    return f"ANSWER:\n{answer or ''}\n\nREFERENCE_SOLUTION:\n{reference_solution}"


def parse_skeleton_response(content: str) -> dict[str, Any]:
    parsed = json.loads(content.strip())
    return normalize_semantic_skeleton(parsed)


def call_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    answer: str | None,
    reference_solution: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_skeleton_compiler_prompt(answer, reference_solution),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


def generate_skeleton_record(
    *,
    problem_id: int,
    example: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    solution = get_solution_text(example)
    ground_truth = get_ground_truth_answer(example)
    last_error = ""
    last_raw = ""
    for attempt in range(max_retries + 1):
        try:
            raw = call_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                answer=ground_truth,
                reference_solution=solution,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            last_raw = raw
            skeleton = parse_skeleton_response(raw)
            return {
                "problem_id": problem_id,
                "ground_truth": ground_truth,
                "skeleton": skeleton,
                "model": model,
                "status": "ok",
            }
        except (json.JSONDecodeError, KeyError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))

    return {
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "skeleton": None,
        "model": model,
        "status": "error",
        "error": last_error,
        "raw_response": last_raw,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic skeleton JSONL from reference solutions.")
    parser.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-indices-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--api-key", type=str, default=os.environ.get("SKELETON_API_KEY"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("SKELETON_BASE_URL"))
    parser.add_argument("--skeleton-model", type=str, default=os.environ.get("SKELETON_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("--api-key or SKELETON_API_KEY is required")
    if not args.base_url:
        raise ValueError("--base-url or SKELETON_BASE_URL is required")

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    indices = read_sample_indices_file(args.sample_indices_file)
    records = [
        generate_skeleton_record(
            problem_id=index,
            example=rows[index],
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.skeleton_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        for index in indices
    ]
    write_jsonl(args.output_file, records)

    failures = [record for record in records if record.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"semantic skeleton generation failed for {len(failures)} examples")


if __name__ == "__main__":
    main()
