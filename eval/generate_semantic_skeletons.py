#!/usr/bin/env python
"""Generate style-neutral semantic skeletons from reference solutions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

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


SKELETON_GENERATION_ERRORS = (
    json.JSONDecodeError,
    KeyError,
    ValueError,
    RuntimeError,
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
    OSError,
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
4. Avoid LaTeX backslash commands in string values. Prefer plain text such as
   frac(a,b), sqrt(x), mod 5, or theta. If a backslash is unavoidable, escape
   it as \\ so the object remains valid JSON.

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


def build_skeleton_compiler_messages(answer: str | None, reference_solution: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_skeleton_compiler_prompt(answer, reference_solution),
        },
    ]


def render_skeleton_compiler_prompt(
    tokenizer: Any,
    *,
    answer: str | None,
    reference_solution: str,
    enable_thinking: bool,
) -> str:
    messages = build_skeleton_compiler_messages(answer, reference_solution)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_json_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def is_valid_unicode_escape_start(content: str, index: int) -> bool:
    if index + 5 >= len(content) or content[index + 1] != "u":
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in content[index + 2 : index + 6])


def escape_latex_backslashes_in_json_strings(content: str) -> str:
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(content):
        ch = content[index]
        if not in_string:
            if ch == '"':
                in_string = True
            output.append(ch)
            index += 1
            continue

        if ch == '"':
            in_string = False
            output.append(ch)
            index += 1
            continue

        if ch != "\\":
            output.append(ch)
            index += 1
            continue

        next_ch = content[index + 1] if index + 1 < len(content) else ""
        if next_ch in {'"', "\\", "/"}:
            output.append(ch)
            output.append(next_ch)
            index += 2
            continue
        if is_valid_unicode_escape_start(content, index):
            output.append(content[index : index + 6])
            index += 6
            continue

        output.append("\\\\")
        index += 1

    return "".join(output)


def contains_control_character(value: Any) -> bool:
    if isinstance(value, str):
        return any(ord(ch) < 32 for ch in value)
    if isinstance(value, list):
        return any(contains_control_character(item) for item in value)
    if isinstance(value, dict):
        return any(contains_control_character(item) for item in value.values())
    return False


def parse_skeleton_response(content: str) -> dict[str, Any]:
    stripped = strip_json_code_fence(content)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        repaired = escape_latex_backslashes_in_json_strings(stripped)
        parsed = json.loads(repaired)
    else:
        if contains_control_character(parsed):
            repaired = escape_latex_backslashes_in_json_strings(stripped)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                pass
    return normalize_semantic_skeleton(parsed)


class VllmSkeletonCompletion:
    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.llm = LLM(
            model=model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            distributed_executor_backend="mp",
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        self.enable_thinking = enable_thinking

    def __call__(self, *, answer: str | None, reference_solution: str) -> str:
        prompt = render_skeleton_compiler_prompt(
            self.tokenizer,
            answer=answer,
            reference_solution=reference_solution,
            enable_thinking=self.enable_thinking,
        )
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no semantic skeleton completion")
        return str(outputs[0].outputs[0].text)


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
        "messages": build_skeleton_compiler_messages(answer, reference_solution),
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
    api_key: str | None,
    base_url: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    skeleton_backend: str = "api",
    completion_fn: Callable[..., str] | None = None,
) -> dict[str, Any]:
    solution = get_solution_text(example)
    ground_truth = get_ground_truth_answer(example)
    if completion_fn is None:
        if not api_key:
            raise ValueError("--api-key or SKELETON_API_KEY is required for --skeleton-backend api")
        if not base_url:
            raise ValueError("--base-url or SKELETON_BASE_URL is required for --skeleton-backend api")

        def completion_fn(*, answer: str | None, reference_solution: str) -> str:
            return call_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                answer=answer,
                reference_solution=reference_solution,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

    last_error = ""
    last_raw = ""
    for attempt in range(max_retries + 1):
        try:
            raw = completion_fn(
                answer=ground_truth,
                reference_solution=solution,
            )
            last_raw = raw
            skeleton = parse_skeleton_response(raw)
            return {
                "problem_id": problem_id,
                "ground_truth": ground_truth,
                "skeleton": skeleton,
                "model": model,
                "skeleton_backend": skeleton_backend,
                "status": "ok",
            }
        except SKELETON_GENERATION_ERRORS as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))

    return {
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "skeleton": None,
        "model": model,
        "skeleton_backend": skeleton_backend,
        "status": "error",
        "error": last_error,
        "raw_response": last_raw,
    }


def generate_skeleton_records(
    *,
    indices: Sequence[int],
    rows: list[dict[str, Any]],
    api_key: str | None,
    base_url: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    skeleton_backend: str = "api",
    completion_fn: Callable[..., str] | None = None,
    api_concurrency: int = 1,
) -> list[dict[str, Any]]:
    total = len(indices)

    def build_record(index: int) -> dict[str, Any]:
        return generate_skeleton_record(
            problem_id=index,
            example=rows[index],
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            skeleton_backend=skeleton_backend,
            completion_fn=completion_fn,
        )

    print(
        f"Generating {total} skeletons with backend={skeleton_backend} concurrency={api_concurrency}",
        file=sys.stderr,
        flush=True,
    )

    if skeleton_backend == "api" and api_concurrency > 1:
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
            for position, record in enumerate(executor.map(build_record, indices), start=1):
                records.append(record)
                if position == 1 or position == total or position % 10 == 0:
                    print(f"Completed {position}/{total} skeletons", file=sys.stderr, flush=True)
        return records

    records = []
    for position, index in enumerate(indices, start=1):
        records.append(build_record(index))
        if position == 1 or position == total or position % 10 == 0:
            print(f"Completed {position}/{total} skeletons", file=sys.stderr, flush=True)
    return records


def iter_skeleton_records(
    *,
    indices: Sequence[int],
    rows: list[dict[str, Any]],
    api_key: str | None,
    base_url: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    skeleton_backend: str = "api",
    completion_fn: Callable[..., str] | None = None,
    api_concurrency: int = 1,
) -> Iterable[dict[str, Any]]:
    total = len(indices)

    def build_record(index: int) -> dict[str, Any]:
        return generate_skeleton_record(
            problem_id=index,
            example=rows[index],
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            skeleton_backend=skeleton_backend,
            completion_fn=completion_fn,
        )

    print(
        f"Generating {total} skeletons with backend={skeleton_backend} concurrency={api_concurrency}",
        file=sys.stderr,
        flush=True,
    )

    if skeleton_backend == "api" and api_concurrency > 1:
        with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
            for position, record in enumerate(executor.map(build_record, indices), start=1):
                if position == 1 or position == total or position % 10 == 0:
                    print(f"Completed {position}/{total} skeletons", file=sys.stderr, flush=True)
                yield record
        return

    for position, index in enumerate(indices, start=1):
        record = build_record(index)
        if position == 1 or position == total or position % 10 == 0:
            print(f"Completed {position}/{total} skeletons", file=sys.stderr, flush=True)
        yield record


def write_jsonl_stream(path: str | Path, records: Iterable[dict[str, Any]], *, flush_every: int = 10) -> None:
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if position % flush_every == 0:
                handle.flush()
        handle.flush()


def load_existing_problem_ids(path: str | Path) -> set[int]:
    existing_ids: set[int] = set()
    output_path = Path(path)
    if not output_path.exists():
        return existing_ids

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            problem_id = record.get("problem_id")
            if isinstance(problem_id, int):
                existing_ids.add(problem_id)
    return existing_ids


def filter_missing_indices(indices: Sequence[int], existing_problem_ids: set[int]) -> list[int]:
    return [index for index in indices if index not in existing_problem_ids]


def resolve_generation_indices(*, row_count: int, sample_indices_file: str | None) -> list[int]:
    if sample_indices_file:
        indices = read_sample_indices_file(sample_indices_file)
    else:
        indices = list(range(row_count))

    invalid_indices = [index for index in indices if index < 0 or index >= row_count]
    if invalid_indices:
        preview = invalid_indices[:5]
        raise ValueError(
            f"sample indices out of range for dataset with {row_count} rows: {preview}"
        )
    return indices


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semantic skeleton JSONL from reference solutions.")
    parser.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--sample-indices-file",
        type=str,
        help="Optional sample-index manifest. If omitted, generate skeletons for every row in the split.",
    )
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument(
        "--skeleton-backend",
        choices=["api", "vllm"],
        default=os.environ.get("SKELETON_BACKEND", "api"),
        help="Backend used to compile reference solutions into semantic skeletons.",
    )
    parser.add_argument("--api-key", type=str, default=os.environ.get("SKELETON_API_KEY"))
    parser.add_argument("--base-url", type=str, default=os.environ.get("SKELETON_BASE_URL"))
    parser.add_argument("--skeleton-model", type=str, default=os.environ.get("SKELETON_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--api-concurrency",
        type=int,
        default=int(os.environ.get("SKELETON_API_CONCURRENCY", "1")),
        help="Parallel API request count used only when --skeleton-backend api.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=int(os.environ.get("SKELETON_FLUSH_EVERY", "10")),
        help="Flush JSONL output after every N records so interrupted runs preserve progress.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start from scratch and overwrite any existing output file instead of resuming.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=int(
            os.environ.get(
                "SKELETON_VLLM_TENSOR_PARALLEL_SIZE",
                os.environ.get("SKELETON_VLLM_TP", "1"),
            )
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("SKELETON_VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=int(os.environ.get("SKELETON_VLLM_MAX_MODEL_LEN", "20000")),
    )
    parser.add_argument("--vllm-top-p", type=float, default=float(os.environ.get("SKELETON_VLLM_TOP_P", "1.0")))
    parser.add_argument("--vllm-top-k", type=int, default=int(os.environ.get("SKELETON_VLLM_TOP_K", "-1")))
    parser.add_argument(
        "--vllm-enable-thinking",
        action="store_true",
        default=_env_flag("SKELETON_VLLM_ENABLE_THINKING", False),
        help="Enable Qwen thinking mode while compiling skeletons. Defaults off to keep JSON-only output stable.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")
    completion_fn: Callable[..., str] | None = None
    if args.skeleton_backend == "api":
        if not args.api_key:
            raise ValueError("--api-key or SKELETON_API_KEY is required")
        if not args.base_url:
            raise ValueError("--base-url or SKELETON_BASE_URL is required")
    else:
        completion_fn = VllmSkeletonCompletion(
            model=args.skeleton_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            top_p=args.vllm_top_p,
            top_k=args.vllm_top_k,
            enable_thinking=args.vllm_enable_thinking,
        )

    from datasets import load_dataset

    print(
        f"Loading dataset {args.dataset}:{args.split}...",
        file=sys.stderr,
        flush=True,
    )
    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    indices = resolve_generation_indices(
        row_count=len(rows),
        sample_indices_file=args.sample_indices_file,
    )
    print(
        f"Loaded {len(rows)} rows; generating {len(indices)} indices",
        file=sys.stderr,
        flush=True,
    )
    resume = not args.no_resume
    existing_problem_ids = load_existing_problem_ids(args.output_file) if resume else set()
    if existing_problem_ids:
        print(
            f"Resuming from {len(existing_problem_ids)} existing records in {args.output_file}",
            file=sys.stderr,
            flush=True,
        )
    remaining_indices = filter_missing_indices(indices, existing_problem_ids)
    if not remaining_indices:
        print("All requested skeletons already exist; nothing to do.", file=sys.stderr, flush=True)
        return

    failures: list[dict[str, Any]] = []
    records = iter_skeleton_records(
        indices=remaining_indices,
        rows=rows,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.skeleton_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        skeleton_backend=args.skeleton_backend,
        completion_fn=completion_fn,
        api_concurrency=args.api_concurrency,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if resume and output_path.exists() else "w"
    with output_path.open(open_mode, encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if position % args.flush_every == 0:
                handle.flush()
            if record.get("status") != "ok":
                failures.append(record)
        handle.flush()

    if failures:
        raise RuntimeError(f"semantic skeleton generation failed for {len(failures)} examples")


if __name__ == "__main__":
    main()
