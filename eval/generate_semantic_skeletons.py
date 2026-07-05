#!/usr/bin/env python
"""Generate style-neutral semantic skeletons from reference solutions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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


NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 422}
MAX_LOGGED_API_RESPONSE_CHARS = 12000


def truncate_for_log(content: str, *, max_chars: int = MAX_LOGGED_API_RESPONSE_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return f"{content[:max_chars]}... [truncated {omitted} chars]"


class SkeletonAPIResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = truncate_for_log(raw_response)
        self.details = details or {}


@dataclass
class ExistingSkeletonSummary:
    ok_records: dict[int, dict[str, Any]] = field(default_factory=dict)
    seen_problem_ids: set[int] = field(default_factory=set)
    record_count: int = 0
    error_count: int = 0
    duplicate_count: int = 0
    invalid_json_count: int = 0


def is_non_retryable_generation_error(error: str) -> bool:
    for status_code in NON_RETRYABLE_HTTP_STATUS_CODES:
        if f"HTTP Error {status_code}" in error:
            return True
    return False


def is_non_retryable_generation_exception(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return int(exc.code) in NON_RETRYABLE_HTTP_STATUS_CODES
    return is_non_retryable_generation_error(str(exc))


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


def build_chat_completion_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def extract_chat_message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("content"), str):
                    parts.append(str(item["content"]))
        return "".join(parts)
    return str(content)


def sorted_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value)


def build_api_response_details(body: Any, choice: Any, message: Any) -> dict[str, Any]:
    details: dict[str, Any] = {
        "body_keys": sorted_keys(body),
        "choice_keys": sorted_keys(choice),
        "message_keys": sorted_keys(message),
    }
    if isinstance(choice, dict) and "finish_reason" in choice:
        details["finish_reason"] = choice.get("finish_reason")
    if isinstance(body, dict) and "usage" in body:
        details["usage"] = body.get("usage")
    return details


def api_response_error(message: str, *, raw_response: str, details: dict[str, Any]) -> SkeletonAPIResponseError:
    return SkeletonAPIResponseError(message, raw_response=raw_response, details=details)


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
    response_format_json: bool = False,
) -> str:
    endpoint = build_chat_completion_endpoint(base_url)
    payload = {
        "model": model,
        "messages": build_skeleton_compiler_messages(answer, reference_solution),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        raise SkeletonAPIResponseError(
            f"HTTP Error {exc.code}: {exc.reason}",
            raw_response=raw_body,
            details={
                "http_status": int(exc.code),
                "http_reason": str(exc.reason),
                "body_keys": [],
                "choice_keys": [],
                "message_keys": [],
            },
        ) from exc

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise api_response_error(
            f"API returned non-JSON response body: {exc}",
            raw_response=raw_body,
            details={"body_keys": [], "choice_keys": [], "message_keys": []},
        ) from exc

    if not isinstance(body, dict):
        raise api_response_error(
            f"API returned JSON {type(body).__name__}, expected object",
            raw_response=raw_body,
            details={"body_keys": [], "choice_keys": [], "message_keys": []},
        )

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise api_response_error(
            "API response does not contain a non-empty choices list",
            raw_response=raw_body,
            details=build_api_response_details(body, {}, {}),
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise api_response_error(
            f"API response choice is {type(choice).__name__}, expected object",
            raw_response=raw_body,
            details=build_api_response_details(body, choice, {}),
        )

    message = choice.get("message")
    if not isinstance(message, dict):
        raise api_response_error(
            "API response choice does not contain a message object",
            raw_response=raw_body,
            details=build_api_response_details(body, choice, message),
        )

    content = extract_chat_message_content(message)
    if not content.strip():
        details = build_api_response_details(body, choice, message)
        finish_reason = details.get("finish_reason")
        raise api_response_error(
            "API returned empty assistant content; "
            f"finish_reason={finish_reason}; "
            f"message_keys={details.get('message_keys', [])}; "
            f"choice_keys={details.get('choice_keys', [])}; "
            f"body_keys={details.get('body_keys', [])}",
            raw_response=raw_body,
            details=details,
        )

    return content


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
    response_format_json: bool = False,
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
                response_format_json=response_format_json,
            )

    last_error = ""
    last_raw = ""
    last_details: dict[str, Any] = {}
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
            if isinstance(exc, SkeletonAPIResponseError):
                last_raw = exc.raw_response
                last_details = dict(exc.details)
            if is_non_retryable_generation_exception(exc):
                break
            if attempt < max_retries:
                time.sleep(min(2**attempt, 8))

    record = {
        "problem_id": problem_id,
        "ground_truth": ground_truth,
        "skeleton": None,
        "model": model,
        "skeleton_backend": skeleton_backend,
        "status": "error",
        "error": last_error,
        "raw_response": last_raw,
    }
    for key, value in last_details.items():
        record[f"api_{key}"] = value
    return record


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
    response_format_json: bool = False,
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
            response_format_json=response_format_json,
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
    retry_until_ok: bool = True,
    retry_delay: float = 2.0,
    max_retry_delay: float = 60.0,
    failure_callback: Callable[[dict[str, Any], int], None] | None = None,
    response_format_json: bool = False,
    abort_after_consecutive_failures: int = 50,
) -> Iterable[dict[str, Any]]:
    total = len(indices)

    def retry_sleep_seconds(failure_round: int) -> float:
        if retry_delay <= 0 or max_retry_delay <= 0:
            return 0.0
        return min(max_retry_delay, retry_delay * (2 ** min(failure_round - 1, 6)))

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
            response_format_json=response_format_json,
        )

    consecutive_failures = 0

    def reset_consecutive_failures() -> None:
        nonlocal consecutive_failures
        consecutive_failures = 0

    def handle_failed_record(record: dict[str, Any], retry_pass: int) -> None:
        nonlocal consecutive_failures
        error = str(record.get("error", ""))
        if is_non_retryable_generation_error(error):
            raise RuntimeError(
                "Encountered non-retryable skeleton generation error for "
                f"problem_id={record.get('problem_id')}: {error}. Check SKELETON_API_KEY, "
                "SKELETON_BASE_URL, SKELETON_MODEL, and endpoint permissions."
            )
        if failure_callback is not None:
            failure_callback(record, retry_pass)
        consecutive_failures += 1
        if (
            abort_after_consecutive_failures > 0
            and consecutive_failures >= abort_after_consecutive_failures
        ):
            raise RuntimeError(
                f"Observed {consecutive_failures} consecutive skeleton generation failures; "
                f"last problem_id={record.get('problem_id')} error={error}. "
                "This usually means the API endpoint is returning empty/non-JSON responses, "
                "the model is incompatible with the requested chat schema, or request settings "
                "need adjustment. Check the failure sidecar for api_* diagnostics. "
                "Set --abort-after-consecutive-failures 0 to disable this guard."
            )

    print(
        f"Generating {total} skeletons with backend={skeleton_backend} concurrency={api_concurrency}",
        file=sys.stderr,
        flush=True,
    )

    pending = list(indices)
    completed = 0
    retry_pass = 0
    while pending:
        retry_pass += 1
        failed_indices: list[int] = []

        if retry_pass > 1:
            delay = retry_sleep_seconds(retry_pass - 1)
            print(
                f"Retry pass {retry_pass} for {len(pending)} deferred skeletons"
                + (f"; sleeping {delay:.1f}s" if delay > 0 else ""),
                file=sys.stderr,
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)

        if skeleton_backend == "api" and api_concurrency > 1:
            with ThreadPoolExecutor(max_workers=api_concurrency) as executor:
                pending_iter = iter(pending)
                future_to_index = {}

                def submit_next() -> None:
                    try:
                        next_index = next(pending_iter)
                    except StopIteration:
                        return
                    future_to_index[executor.submit(build_record, next_index)] = next_index

                for _ in range(api_concurrency):
                    submit_next()

                while future_to_index:
                    for future in as_completed(future_to_index):
                        break
                    index = future_to_index[future]
                    del future_to_index[future]
                    record = future.result()
                    if record.get("status") == "ok" or not retry_until_ok:
                        reset_consecutive_failures()
                        completed += 1
                        if completed == 1 or completed == total or completed % 10 == 0:
                            print(f"Completed {completed}/{total} skeletons", file=sys.stderr, flush=True)
                        yield record
                    else:
                        handle_failed_record(record, retry_pass)
                        failed_indices.append(index)
                    submit_next()
        else:
            for index in pending:
                record = build_record(index)
                if record.get("status") == "ok" or not retry_until_ok:
                    reset_consecutive_failures()
                    completed += 1
                    if completed == 1 or completed == total or completed % 10 == 0:
                        print(f"Completed {completed}/{total} skeletons", file=sys.stderr, flush=True)
                    yield record
                else:
                    handle_failed_record(record, retry_pass)
                    failed_indices.append(index)

        if failed_indices and retry_until_ok:
            print(
                f"Deferred {len(failed_indices)}/{len(pending)} failed skeletons to a later pass",
                file=sys.stderr,
                flush=True,
            )
        pending = failed_indices


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


def default_failure_file_for_output(output_file: str | Path) -> Path:
    output_path = Path(output_file)
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.failures{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.failures.jsonl")


def load_existing_skeleton_summary(path: str | Path) -> ExistingSkeletonSummary:
    summary = ExistingSkeletonSummary()
    output_path = Path(path)
    if not output_path.exists():
        return summary

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                summary.invalid_json_count += 1
                continue

            summary.record_count += 1
            problem_id = record.get("problem_id")
            if not isinstance(problem_id, int):
                continue

            if problem_id in summary.seen_problem_ids:
                summary.duplicate_count += 1
            summary.seen_problem_ids.add(problem_id)

            status = str(record.get("status", "ok")).lower()
            if status in {"ok", "success"}:
                summary.ok_records.setdefault(problem_id, record)
            else:
                summary.error_count += 1
    return summary


def load_existing_problem_ids(path: str | Path) -> set[int]:
    return load_existing_skeleton_summary(path).seen_problem_ids


def filter_missing_indices(indices: Sequence[int], existing_problem_ids: set[int]) -> list[int]:
    return [index for index in indices if index not in existing_problem_ids]


def filter_pending_indices(
    indices: Sequence[int],
    *,
    existing_ok_problem_ids: set[int],
    existing_seen_problem_ids: set[int],
) -> list[int]:
    missing = [index for index in indices if index not in existing_ok_problem_ids]
    if not existing_seen_problem_ids:
        return missing

    resume_after = max(existing_seen_problem_ids)
    forward = [index for index in missing if index > resume_after]
    repair = [index for index in missing if index <= resume_after]
    return forward + repair


def rewrite_clean_output_file(
    path: str | Path,
    ok_records: dict[int, dict[str, Any]],
    indices: Sequence[int],
) -> None:
    write_jsonl(path, (ok_records[index] for index in indices if index in ok_records))


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
        "--allow-error-records",
        action="store_true",
        help="After --max-retries is exhausted, write error records instead of retrying each problem until ok.",
    )
    parser.add_argument(
        "--failure-file",
        type=str,
        default=os.environ.get("SKELETON_FAILURE_FILE"),
        help="Sidecar JSONL for failed attempts. Defaults to <output-stem>.failures.jsonl.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.environ.get("SKELETON_RETRY_DELAY", "2.0")),
        help="Initial delay between failed retry batches when retrying until ok.",
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=float(os.environ.get("SKELETON_MAX_RETRY_DELAY", "60.0")),
        help="Maximum delay between failed retry batches when retrying until ok.",
    )
    parser.add_argument(
        "--abort-after-consecutive-failures",
        type=int,
        default=int(os.environ.get("SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES", "50")),
        help=(
            "Abort after this many consecutive retryable failures, which usually indicates "
            "an API/config issue. Set 0 to retry forever."
        ),
    )
    parser.add_argument(
        "--response-format-json",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SKELETON_RESPONSE_FORMAT_JSON", False),
        help="Request OpenAI-compatible JSON object mode via response_format={type: json_object}.",
    )
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
    if args.retry_delay < 0:
        raise ValueError("--retry-delay must be non-negative")
    if args.max_retry_delay < 0:
        raise ValueError("--max-retry-delay must be non-negative")
    if args.abort_after_consecutive_failures < 0:
        raise ValueError("--abort-after-consecutive-failures must be non-negative")
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
    output_path = Path(args.output_file)
    existing_summary = (
        load_existing_skeleton_summary(output_path)
        if resume
        else ExistingSkeletonSummary()
    )
    existing_ok_problem_ids = set(existing_summary.ok_records)
    existing_seen_problem_ids = existing_summary.seen_problem_ids
    if existing_seen_problem_ids:
        print(
            "Resuming from "
            f"{len(existing_ok_problem_ids)} ok records "
            f"({existing_summary.error_count} errors, "
            f"{existing_summary.duplicate_count} duplicates, "
            f"{existing_summary.invalid_json_count} invalid JSON lines) "
            f"in {args.output_file}",
            file=sys.stderr,
            flush=True,
        )
    dirty_existing_output = (
        existing_summary.error_count > 0
        or existing_summary.duplicate_count > 0
        or existing_summary.invalid_json_count > 0
    )
    if resume and output_path.exists() and dirty_existing_output:
        rewrite_clean_output_file(output_path, existing_summary.ok_records, indices)
        print(
            f"Cleaned existing output to {len(existing_ok_problem_ids)} unique ok records",
            file=sys.stderr,
            flush=True,
        )

    remaining_indices = filter_pending_indices(
        indices,
        existing_ok_problem_ids=existing_ok_problem_ids,
        existing_seen_problem_ids=existing_seen_problem_ids,
    )
    if not remaining_indices:
        print("All requested skeletons already exist; nothing to do.", file=sys.stderr, flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = Path(args.failure_file) if args.failure_file else default_failure_file_for_output(output_path)
    failure_handle = None
    if not args.allow_error_records:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_handle = failure_path.open("a", encoding="utf-8")
        print(f"Failure sidecar: {failure_path}", file=sys.stderr, flush=True)

        def record_failure(record: dict[str, Any], retry_pass: int) -> None:
            payload = dict(record)
            payload["retry_pass"] = retry_pass
            payload["attempted_at_unix"] = time.time()
            failure_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            failure_handle.flush()

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
            retry_until_ok=True,
            retry_delay=args.retry_delay,
            max_retry_delay=args.max_retry_delay,
            failure_callback=record_failure,
            response_format_json=args.response_format_json,
            abort_after_consecutive_failures=args.abort_after_consecutive_failures,
        )
    else:
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
            retry_until_ok=False,
            retry_delay=args.retry_delay,
            max_retry_delay=args.max_retry_delay,
            response_format_json=args.response_format_json,
            abort_after_consecutive_failures=args.abort_after_consecutive_failures,
        )
    open_mode = "a" if resume and output_path.exists() else "w"
    failures: list[dict[str, Any]] = []
    try:
        with output_path.open(open_mode, encoding="utf-8") as handle:
            for position, record in enumerate(records, start=1):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if position % args.flush_every == 0:
                    handle.flush()
                if record.get("status") != "ok":
                    failures.append(record)
            handle.flush()
    finally:
        if failure_handle is not None:
            failure_handle.close()

    if failures:
        raise RuntimeError(f"semantic skeleton generation failed for {len(failures)} examples")


if __name__ == "__main__":
    main()
