#!/usr/bin/env python
"""Generate privileged-teacher continuations at the largest KL positions."""

from __future__ import annotations

import argparse
import heapq
import html
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

try:
    from .quick_jsonl_merge import iter_jsonl_records
    from .quick_logit_probe import context_prompt_ids_for_condition, lora_adapter_exists
    from .quick_opsd_common import read_skeleton_file, shard_items, write_json, write_jsonl
except ImportError:  # pragma: no cover
    from quick_jsonl_merge import iter_jsonl_records
    from quick_logit_probe import context_prompt_ids_for_condition, lora_adapter_exists
    from quick_opsd_common import read_skeleton_file, shard_items, write_json, write_jsonl


REFERENCE_LABEL = "reference"
SKELETON_LABEL = "skeleton"
TEACHER_CONDITIONS = ("teacher_reference", "teacher_skeleton")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue reference and skeleton teachers at global high-KL student positions."
    )
    parser.add_argument("--model", default="/data0/shared/Qwen3-1.7B")
    parser.add_argument("--base-model", dest="base_model")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--kl-file")
    parser.add_argument("--student-rollout-file")
    parser.add_argument("--skeleton-file")
    parser.add_argument("--output-file")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=20000)
    parser.add_argument("--context-snippet-tokens", type=int, default=48)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--hf-device-map", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--input-file", help="Merged continuation JSONL used by --render-only.")
    parser.add_argument("--summary-file")
    parser.add_argument("--report-file")
    args = parser.parse_args()

    if args.base_model:
        args.model = args.base_model
    else:
        args.base_model = args.model

    if args.top_n <= 0:
        parser.error("--top-n must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_context_tokens <= 0:
        parser.error("--max-context-tokens must be positive")

    if args.render_only:
        if not args.input_file or not args.summary_file or not args.report_file:
            parser.error("--render-only requires --input-file, --summary-file, and --report-file")
    else:
        missing = [
            name
            for name, value in (
                ("--kl-file", args.kl_file),
                ("--student-rollout-file", args.student_rollout_file),
                ("--skeleton-file", args.skeleton_file),
                ("--output-file", args.output_file),
            )
            if not value
        ]
        if missing:
            parser.error(f"worker mode requires {', '.join(missing)}")
    return args


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _token_ids(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(token_id) for token_id in value]


def _contrast_label(contrast: Any) -> str | None:
    name = str(contrast or "")
    if name.startswith("teacher_reference_vs_"):
        return REFERENCE_LABEL
    if name.startswith("teacher_skeleton_vs_"):
        return SKELETON_LABEL
    return None


def _record_case_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("problem_id")),
        _as_int(record.get("target_sample_index", record.get("sample_index", 0))),
        str(record.get("target_condition") or "student"),
    )


def _spike_key(record: dict[str, Any], position: int) -> tuple[str, int, str, int]:
    return (*_record_case_key(record), int(position))


def _value_at(record: dict[str, Any], field: str, position: int) -> float | None:
    values = record.get(field)
    if not isinstance(values, list) or position >= len(values):
        return None
    return float(values[position])


def _top_distribution_at(record: dict[str, Any], position: int) -> dict[str, Any]:
    rows = record.get("top_kl_positions")
    if not isinstance(rows, list):
        return {"teacher_top_tokens": [], "base_top_tokens": []}
    for row in rows:
        if _as_int(row.get("position"), -1) == position:
            return {
                "teacher_top_tokens": row.get("teacher_top_tokens") or [],
                "base_top_tokens": row.get("base_top_tokens") or [],
            }
    return {"teacher_top_tokens": [], "base_top_tokens": []}


def select_global_spikes(
    records_factory: Callable[[], Iterable[dict[str, Any]]],
    top_n: int,
) -> list[dict[str, Any]]:
    """Select global unique positions and hydrate both teacher contrasts."""

    if top_n <= 0:
        return []
    candidates: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for record in records_factory():
        label = _contrast_label(record.get("contrast"))
        kl_values = record.get("kl_per_token")
        if record.get("record_type") != "kl_contrast" or label is None or not isinstance(kl_values, list):
            continue
        local_candidates = heapq.nlargest(
            min(top_n, len(kl_values)),
            enumerate(kl_values),
            key=lambda item: float(item[1]),
        )
        token_texts = record.get("token_texts") if isinstance(record.get("token_texts"), list) else []
        for position, kl_value in local_candidates:
            key = _spike_key(record, position)
            row = candidates.setdefault(
                key,
                {
                    "case_id": record.get("case_id"),
                    "problem_id": key[0],
                    "sample_index": key[1],
                    "target_condition": key[2],
                    "position": key[3],
                    "student_token_text": token_texts[position] if position < len(token_texts) else "",
                    "max_kl": float(kl_value),
                },
            )
            row["max_kl"] = max(float(row["max_kl"]), float(kl_value))

    selected = sorted(
        candidates.values(),
        key=lambda row: (
            -float(row["max_kl"]),
            str(row["problem_id"]),
            int(row["sample_index"]),
            str(row["target_condition"]),
            int(row["position"]),
        ),
    )[:top_n]
    selected_by_case: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        row["contrast_metrics"] = {}
        selected_by_case[(row["problem_id"], row["sample_index"], row["target_condition"])][
            row["position"]
        ] = row

    for record in records_factory():
        label = _contrast_label(record.get("contrast"))
        if label is None:
            continue
        selected_positions = selected_by_case.get(_record_case_key(record))
        if not selected_positions:
            continue
        token_texts = record.get("token_texts") if isinstance(record.get("token_texts"), list) else []
        for position, row in selected_positions.items():
            kl_value = _value_at(record, "kl_per_token", position)
            if kl_value is None:
                continue
            distribution = _top_distribution_at(record, position)
            row["contrast_metrics"][label] = {
                "contrast": record.get("contrast"),
                "kl": kl_value,
                "delta_logp_target": _value_at(record, "delta_logp_target_per_token", position),
                "teacher_entropy": _value_at(record, "teacher_entropy_per_token", position),
                "student_entropy": _value_at(record, "student_entropy_per_token", position),
                **distribution,
            }
            if position < len(token_texts):
                row["student_token_text"] = token_texts[position]

    for row in selected:
        labels = set(row["contrast_metrics"])
        if labels != {REFERENCE_LABEL, SKELETON_LABEL}:
            raise ValueError(
                "Every selected KL spike requires both reference and skeleton contrast records; "
                f"missing for problem={row['problem_id']} sample={row['sample_index']} "
                f"position={row['position']}"
            )
        row["reference_kl"] = float(row["contrast_metrics"][REFERENCE_LABEL]["kl"])
        row["skeleton_kl"] = float(row["contrast_metrics"][SKELETON_LABEL]["kl"])
        row["max_kl"] = max(row["reference_kl"], row["skeleton_kl"])

    selected.sort(
        key=lambda row: (
            -float(row["max_kl"]),
            str(row["problem_id"]),
            int(row["sample_index"]),
            int(row["position"]),
        )
    )
    for rank, row in enumerate(selected, 1):
        row["rank"] = rank
    return selected


def build_generation_input_ids(
    prompt_ids: list[int],
    completion_ids: list[int],
    position: int,
    max_new_tokens: int,
    max_context_tokens: int,
) -> list[int]:
    """Build prompt + student prefix, deliberately excluding token at position."""

    if position < 0 or position >= len(completion_ids):
        raise ValueError(
            f"KL position {position} is outside completion_token_ids with length {len(completion_ids)}"
        )
    input_ids = list(prompt_ids) + list(completion_ids[:position])
    if len(input_ids) + max_new_tokens > max_context_tokens:
        raise ValueError(
            "Teacher continuation input exceeds max context; refusing to truncate privileged prompt "
            f"or student prefix ({len(input_ids)} + {max_new_tokens} > {max_context_tokens})"
        )
    return input_ids


def prepare_spike_case(
    spike: dict[str, Any],
    rollout: dict[str, Any],
    tokenizer: Any,
    display_tokens: int,
    context_tokens: int = 48,
) -> dict[str, Any]:
    completion_ids = _token_ids(rollout.get("completion_token_ids"))
    position = _as_int(spike.get("position"), -1)
    if position < 0 or position >= len(completion_ids):
        raise ValueError(
            f"KL position {position} is outside completion_token_ids for problem={spike.get('problem_id')}"
        )
    suffix_ids = completion_ids[position : position + max(0, display_tokens)]
    context_ids = completion_ids[max(0, position - max(0, context_tokens)) : position]
    return {
        **rollout,
        **spike,
        "completion_token_ids": completion_ids,
        "student_prefix_token_count": position,
        "student_token_text": tokenizer.decode([completion_ids[position]], skip_special_tokens=False),
        "student_suffix_token_ids": suffix_ids,
        "student_suffix_text": tokenizer.decode(suffix_ids, skip_special_tokens=False),
        "context_before_token_ids": context_ids,
        "context_before_text": tokenizer.decode(context_ids, skip_special_tokens=False),
        "reference_solution": str(rollout.get("solution") or ""),
    }


def teacher_input_ids_for_case(
    tokenizer: Any,
    case: dict[str, Any],
    condition: str,
    skeletons: dict[int, dict[str, Any]],
    max_new_tokens: int,
    max_context_tokens: int,
) -> tuple[list[int], int, str]:
    prompt_ids, prompt_source = context_prompt_ids_for_condition(
        tokenizer=tokenizer,
        case=case,
        condition=condition,
        skeletons=skeletons,
    )
    completion_ids = _token_ids(case.get("completion_token_ids"))
    input_ids = build_generation_input_ids(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        position=_as_int(case.get("position"), -1),
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
    )
    return input_ids, len(prompt_ids), prompt_source


def _load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.hf_device_map == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--hf-device-map cuda requires CUDA")
        model_kwargs["device_map"] = {"": "cuda:0"}
    elif args.hf_device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.checkpoint_dir:
        if not lora_adapter_exists(args.checkpoint_dir):
            raise ValueError(
                "--checkpoint-dir does not contain adapter_model.safetensors or adapter_model.bin: "
                f"{args.checkpoint_dir}"
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()
    return model, tokenizer


def generate_greedy_continuation(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id or pad_token_id for generation")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    new_token_ids = [int(token_id) for token_id in generated[0, input_tensor.shape[1] :].tolist()]
    stopped_on_eos = eos_token_id is not None and bool(new_token_ids) and new_token_ids[-1] == eos_token_id
    return {
        "token_ids": new_token_ids,
        "text": tokenizer.decode(new_token_ids, skip_special_tokens=False),
        "clean_text": tokenizer.decode(new_token_ids, skip_special_tokens=True),
        "num_tokens": len(new_token_ids),
        "finish_reason": "stop" if stopped_on_eos else "length",
        "first_token_text": (
            tokenizer.decode(new_token_ids[:1], skip_special_tokens=False) if new_token_ids else ""
        ),
    }


def _rollout_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("problem_id")),
        _as_int(record.get("sample_index", 0)),
        str(record.get("condition") or "student"),
    )


def run_worker(args: argparse.Namespace) -> list[dict[str, Any]]:
    records_factory = lambda: iter_jsonl_records(args.kl_file)
    spikes = select_global_spikes(records_factory, args.top_n)
    spikes = shard_items(spikes, args.shard_id, args.num_shards)
    if not spikes:
        write_jsonl(args.output_file, [])
        return []

    rollouts = {
        _rollout_key(record): record
        for record in iter_jsonl_records(args.student_rollout_file)
        if str(record.get("condition") or "") == "student"
    }
    skeletons = read_skeleton_file(args.skeleton_file)
    model, tokenizer = _load_model_and_tokenizer(args)
    output_records: list[dict[str, Any]] = []

    for spike in spikes:
        key = (str(spike["problem_id"]), int(spike["sample_index"]), str(spike["target_condition"]))
        rollout = rollouts.get(key)
        if rollout is None:
            raise ValueError(f"Missing student rollout for spike key {key}")
        case = prepare_spike_case(
            spike=spike,
            rollout=rollout,
            tokenizer=tokenizer,
            display_tokens=args.max_new_tokens,
            context_tokens=args.context_snippet_tokens,
        )
        problem_id = _as_int(case.get("problem_id"), -1)
        if problem_id not in skeletons:
            raise ValueError(f"Missing semantic skeleton for problem_id={problem_id}")
        case["semantic_skeleton"] = skeletons[problem_id]
        case["continuations"] = {}
        for condition in TEACHER_CONDITIONS:
            input_ids, prompt_count, prompt_source = teacher_input_ids_for_case(
                tokenizer=tokenizer,
                case=case,
                condition=condition,
                skeletons=skeletons,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
            )
            generated = generate_greedy_continuation(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )
            case["continuations"][condition] = {
                **generated,
                "prompt_token_count": prompt_count,
                "student_prefix_token_count": int(case["position"]),
                "input_token_count": len(input_ids),
                "prompt_token_source": prompt_source,
            }
        case["generation_config"] = {
            "base_model": args.base_model,
            "checkpoint_dir": args.checkpoint_dir,
            "decoding": "greedy",
            "top_n": args.top_n,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
        }
        output_records.append(case)

    write_jsonl(args.output_file, output_records)
    return output_records


def _escaped_pre(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_html_report(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in sorted(records, key=lambda row: int(row.get("rank", 0))):
        continuations = record.get("continuations") or {}
        reference = continuations.get("teacher_reference") or {}
        skeleton = continuations.get("teacher_skeleton") or {}
        metrics_json = json.dumps(record.get("contrast_metrics") or {}, ensure_ascii=False, indent=2)
        skeleton_json = json.dumps(record.get("semantic_skeleton") or {}, ensure_ascii=False, indent=2)
        cards.append(
            f"""
            <section class="card">
              <h2>Rank {int(record.get('rank', 0))} · problem {_escaped_pre(record.get('problem_id'))}
                · sample {int(record.get('sample_index', 0))} · position {int(record.get('position', 0))}</h2>
              <div class="metrics">max KL {float(record.get('max_kl', 0.0)):.6f} ·
                reference KL {float(record.get('reference_kl', 0.0)):.6f} ·
                skeleton KL {float(record.get('skeleton_kl', 0.0)):.6f}</div>
              <details><summary>Problem</summary><pre>{_escaped_pre(record.get('problem'))}</pre></details>
              <div class="branch"><pre>{_escaped_pre(record.get('context_before_text'))}<mark>▌ branch before {_escaped_pre(record.get('student_token_text'))}</mark></pre></div>
              <div class="columns">
                <article><h3>Student original</h3><pre>{_escaped_pre(record.get('student_suffix_text'))}</pre></article>
                <article><h3>Reference teacher</h3><pre>{_escaped_pre(reference.get('text'))}</pre></article>
                <article><h3>Skeleton teacher</h3><pre>{_escaped_pre(skeleton.get('text'))}</pre></article>
              </div>
              <details><summary>Reference solution</summary><pre>{_escaped_pre(record.get('reference_solution'))}</pre></details>
              <details><summary>Semantic skeleton</summary><pre>{_escaped_pre(skeleton_json)}</pre></details>
              <details><summary>Top distributions and per-position metrics</summary><pre>{_escaped_pre(metrics_json)}</pre></details>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher KL spike continuations</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f6f8;color:#202124;margin:0;padding:24px}}
main{{max-width:1500px;margin:auto}} .card{{background:white;border:1px solid #d9dde3;border-radius:12px;padding:18px;margin:0 0 20px}}
h1,h2,h3{{margin-top:0}} .metrics{{color:#4b5563;margin-bottom:12px}} .columns{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:14px 0}}
article{{border:1px solid #d9dde3;border-radius:8px;padding:12px;min-width:0}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0;font-family:ui-monospace,monospace}}
.branch{{background:#f7f3ff;border-left:4px solid #7c3aed;padding:8px 12px}} mark{{background:#fde68a}} details{{margin-top:10px}}
@media(max-width:900px){{.columns{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Teacher KL spike continuations</h1>
<p>Each teacher branches before the highlighted student token and uses greedy decoding.</p>
{''.join(cards)}</main></body></html>"""


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_records = sorted(records, key=lambda row: int(row.get("rank", 0)))
    return {
        "num_records": len(sorted_records),
        "num_successful_records": len(sorted_records),
        "num_failed_records": 0,
        "ranks": [int(row.get("rank", 0)) for row in sorted_records],
        "generation_config": sorted_records[0].get("generation_config") if sorted_records else {},
        "conditions": list(TEACHER_CONDITIONS),
    }


def render_outputs(args: argparse.Namespace) -> None:
    records = list(iter_jsonl_records(args.input_file))
    records.sort(key=lambda row: int(row.get("rank", 0)))
    write_json(args.summary_file, summarize_records(records))
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_html_report(records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.render_only:
        render_outputs(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
