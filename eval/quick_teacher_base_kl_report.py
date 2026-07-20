#!/usr/bin/env python3
"""Build legacy-compatible reports for teacher-base KL contrasts."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .quick_opsd_common import read_jsonl, read_skeleton_file
except ImportError:  # pragma: no cover - used when run as python eval/script.py
    from quick_opsd_common import read_jsonl, read_skeleton_file


REFERENCE_CONTRAST = "teacher_reference_vs_teacher_base"
SKELETON_CONTRAST = "teacher_skeleton_vs_teacher_base"
TARGET_CONDITION = "teacher_base"
CSV_FIELDS = (
    "case_id",
    "problem_id",
    "sample_index",
    "target_condition",
    "position",
    "token",
    "reference_kl",
    "skeleton_kl",
    "kl_diff_skeleton_minus_reference",
    "abs_kl_diff",
    "max_kl",
    "reference_delta_logp",
    "skeleton_delta_logp",
    "reference_delta_entropy",
    "skeleton_delta_entropy",
    "reference_teacher_entropy",
    "skeleton_teacher_entropy",
    "base_entropy",
    "saved_for_reference",
    "saved_for_skeleton",
    "snippet",
    "reference_teacher_top8",
    "reference_base_top8",
    "skeleton_teacher_top8",
    "skeleton_base_top8",
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_at(values: Any, position: int) -> float:
    if not isinstance(values, list) or position < 0 or position >= len(values):
        return 0.0
    return float(values[position])


def _rollout_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("problem_id")),
        _as_int(record.get("sample_index", 0)),
        str(record.get("condition") or ""),
    )


def _contrast_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("problem_id")),
        _as_int(record.get("target_sample_index", record.get("sample_index", 0))),
        str(record.get("target_condition") or ""),
    )


def _top_positions(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = record.get("top_kl_positions")
    if not isinstance(rows, list):
        return {}
    return {
        _as_int(row.get("position"), -1): row
        for row in rows
        if isinstance(row, dict) and _as_int(row.get("position"), -1) >= 0
    }


def _validate_token_arrays(
    case_id: str,
    reference: dict[str, Any],
    skeleton: dict[str, Any],
) -> int:
    fields = (
        "token_texts",
        "kl_per_token",
        "delta_logp_target_per_token",
        "teacher_entropy_per_token",
        "student_entropy_per_token",
    )
    lengths = {
        f"reference.{field}": len(reference.get(field, []))
        for field in fields
        if isinstance(reference.get(field), list)
    }
    lengths.update(
        {
            f"skeleton.{field}": len(skeleton.get(field, []))
            for field in fields
            if isinstance(skeleton.get(field), list)
        }
    )
    expected_entries = len(fields) * 2
    if len(lengths) != expected_entries or len(set(lengths.values())) != 1:
        raise ValueError(f"Mismatched token array lengths for case {case_id}: {lengths}")
    token_count = next(iter(lengths.values()))
    if reference["token_texts"] != skeleton["token_texts"]:
        raise ValueError(f"Mismatched token text arrays for case {case_id}")
    return token_count


def build_teacher_base_cases(
    logit_records: Iterable[dict[str, Any]],
    rollout_records: Iterable[dict[str, Any]],
    skeletons: dict[int, dict[str, Any]] | dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair reference/skeleton KL records for every teacher-base trajectory."""

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in logit_records:
        contrast = str(record.get("contrast") or "")
        if contrast not in {REFERENCE_CONTRAST, SKELETON_CONTRAST}:
            continue
        case_id = str(record.get("case_id") or "")
        if not case_id:
            raise ValueError(f"KL record for {contrast} is missing case_id")
        if contrast in grouped[case_id]:
            raise ValueError(f"Duplicate {contrast} record for case {case_id}")
        grouped[case_id][contrast] = record

    rollout_by_key = {
        _rollout_key(record): record
        for record in rollout_records
        if str(record.get("condition") or "") == TARGET_CONDITION
    }
    cases: list[dict[str, Any]] = []
    for case_id, contrasts in sorted(grouped.items()):
        if set(contrasts) != {REFERENCE_CONTRAST, SKELETON_CONTRAST}:
            raise ValueError(
                f"Teacher-base report requires both reference and skeleton contrasts for case {case_id}"
            )
        reference = contrasts[REFERENCE_CONTRAST]
        skeleton = contrasts[SKELETON_CONTRAST]
        if _contrast_key(reference) != _contrast_key(skeleton):
            raise ValueError(f"Teacher contrast keys do not match for case {case_id}")
        key = _contrast_key(reference)
        if key[2] != TARGET_CONDITION:
            raise ValueError(f"Teacher-base report received target condition {key[2]!r} for case {case_id}")
        rollout = rollout_by_key.get(key)
        if rollout is None:
            raise ValueError(f"Missing teacher_base rollout for case {case_id} and key {key}")
        token_count = _validate_token_arrays(case_id, reference, skeleton)
        try:
            problem_id = int(key[0])
        except ValueError:
            problem_id = key[0]
        semantic_skeleton = skeletons.get(problem_id, skeletons.get(str(problem_id), {}))
        cases.append(
            {
                "case_id": case_id,
                "problem_id": problem_id,
                "sample_index": key[1],
                "target_condition": TARGET_CONDITION,
                "problem": str(rollout.get("problem") or ""),
                "ground_truth": rollout.get("ground_truth"),
                "predicted_answer": rollout.get("predicted_answer"),
                "correct": rollout.get("correct"),
                "formatted": rollout.get("formatted"),
                "finish_reason": rollout.get("finish_reason"),
                "completion_tokens": rollout.get("completion_tokens", token_count),
                "full_generation": str(rollout.get("full_generation") or ""),
                "semantic_skeleton": semantic_skeleton or {},
                "num_tokens": token_count,
                "tokens": list(reference["token_texts"]),
                "reference_kl": [float(value) for value in reference["kl_per_token"]],
                "skeleton_kl": [float(value) for value in skeleton["kl_per_token"]],
                "reference_delta_logp": [
                    float(value) for value in reference["delta_logp_target_per_token"]
                ],
                "skeleton_delta_logp": [
                    float(value) for value in skeleton["delta_logp_target_per_token"]
                ],
                "reference_teacher_entropy": [
                    float(value) for value in reference["teacher_entropy_per_token"]
                ],
                "skeleton_teacher_entropy": [
                    float(value) for value in skeleton["teacher_entropy_per_token"]
                ],
                "base_entropy": [
                    float(value) for value in reference["student_entropy_per_token"]
                ],
                "reference_top_positions": _top_positions(reference),
                "skeleton_top_positions": _top_positions(skeleton),
                "reference_mean_kl": float(reference.get("mean_kl", 0.0)),
                "skeleton_mean_kl": float(skeleton.get("mean_kl", 0.0)),
                "reference_top1_agreement": float(reference.get("top1_agreement", 0.0)),
                "skeleton_top1_agreement": float(skeleton.get("top1_agreement", 0.0)),
                "reference_topk_jaccard": float(reference.get("topk_jaccard", 0.0)),
                "skeleton_topk_jaccard": float(skeleton.get("topk_jaccard", 0.0)),
                "reference_mean_delta_logp": float(
                    reference.get("mean_delta_logp_target", 0.0)
                ),
                "skeleton_mean_delta_logp": float(
                    skeleton.get("mean_delta_logp_target", 0.0)
                ),
                "reference_mean_delta_entropy": float(reference.get("mean_delta_entropy", 0.0)),
                "skeleton_mean_delta_entropy": float(skeleton.get("mean_delta_entropy", 0.0)),
            }
        )
    return cases


def _snippet(tokens: list[str], position: int, radius: int = 12) -> str:
    before = "".join(tokens[max(0, position - radius) : position])
    token = tokens[position]
    after = "".join(tokens[position + 1 : position + radius + 1])
    return f"{before}[[{token}]]{after}".replace("\n", "⏎")


def build_spike_rows(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one legacy-compatible row per saved top-KL token position."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        reference_top = case.get("reference_top_positions") or {}
        skeleton_top = case.get("skeleton_top_positions") or {}
        positions = sorted(set(reference_top) | set(skeleton_top))
        for position in positions:
            tokens = case["tokens"]
            if position < 0 or position >= len(tokens):
                raise ValueError(
                    f"Top-KL position {position} is outside token array for case {case['case_id']}"
                )
            reference_kl = _as_float_at(case["reference_kl"], position)
            skeleton_kl = _as_float_at(case["skeleton_kl"], position)
            reference_dist = reference_top.get(position, {})
            skeleton_dist = skeleton_top.get(position, {})
            rows.append(
                {
                    "case_id": case["case_id"],
                    "problem_id": case["problem_id"],
                    "sample_index": case["sample_index"],
                    "target_condition": TARGET_CONDITION,
                    "position": position,
                    "token": tokens[position],
                    "reference_kl": reference_kl,
                    "skeleton_kl": skeleton_kl,
                    "kl_diff_skeleton_minus_reference": skeleton_kl - reference_kl,
                    "abs_kl_diff": abs(skeleton_kl - reference_kl),
                    "max_kl": max(reference_kl, skeleton_kl),
                    "reference_delta_logp": _as_float_at(case["reference_delta_logp"], position),
                    "skeleton_delta_logp": _as_float_at(case["skeleton_delta_logp"], position),
                    "reference_delta_entropy": (
                        _as_float_at(case["reference_teacher_entropy"], position)
                        - _as_float_at(case["base_entropy"], position)
                    ),
                    "skeleton_delta_entropy": (
                        _as_float_at(case["skeleton_teacher_entropy"], position)
                        - _as_float_at(case["base_entropy"], position)
                    ),
                    "reference_teacher_entropy": _as_float_at(
                        case["reference_teacher_entropy"], position
                    ),
                    "skeleton_teacher_entropy": _as_float_at(
                        case["skeleton_teacher_entropy"], position
                    ),
                    "base_entropy": _as_float_at(case["base_entropy"], position),
                    "snippet": _snippet(tokens, position),
                    "reference_teacher_top_tokens": reference_dist.get("teacher_top_tokens") or [],
                    "reference_base_top_tokens": reference_dist.get("base_top_tokens") or [],
                    "skeleton_teacher_top_tokens": skeleton_dist.get("teacher_top_tokens") or [],
                    "skeleton_base_top_tokens": skeleton_dist.get("base_top_tokens") or [],
                    "saved_for_reference": position in reference_top,
                    "saved_for_skeleton": position in skeleton_top,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["max_kl"]),
            str(row["problem_id"]),
            int(row["sample_index"]),
            int(row["position"]),
        )
    )
    return rows


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _top_tokens_text(rows: Any, limit: int = 8) -> str:
    if not isinstance(rows, list):
        return ""
    formatted = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        formatted.append(f"{str(row.get('token') or '')!r}:{float(row.get('prob', 0.0)):.6g}")
    return " | ".join(formatted)


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: row.get(field) for field in CSV_FIELDS},
        "reference_teacher_top8": _top_tokens_text(row.get("reference_teacher_top_tokens")),
        "reference_base_top8": _top_tokens_text(row.get("reference_base_top_tokens")),
        "skeleton_teacher_top8": _top_tokens_text(row.get("skeleton_teacher_top_tokens")),
        "skeleton_base_top8": _top_tokens_text(row.get("skeleton_base_top_tokens")),
    }


def _report_summary(
    cases: list[dict[str, Any]],
    rollout_summary: dict[str, Any],
    spikes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "num_cases": len(cases),
        "reference_contrast": REFERENCE_CONTRAST,
        "skeleton_contrast": SKELETON_CONTRAST,
        "target_condition": TARGET_CONDITION,
        "mean_reference_kl": _mean(case.get("reference_mean_kl") for case in cases),
        "mean_skeleton_kl": _mean(case.get("skeleton_mean_kl") for case in cases),
        "mean_kl_diff_skeleton_minus_reference": _mean(
            float(case.get("skeleton_mean_kl", 0.0)) - float(case.get("reference_mean_kl", 0.0))
            for case in cases
        ),
        "max_token_kl": max((float(row.get("max_kl", 0.0)) for row in spikes), default=0.0),
        "rollout_summary": rollout_summary,
    }


def _metric_table(rollout_summary: dict[str, Any]) -> str:
    conditions = rollout_summary.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        return "<p>No rollout performance summary was provided.</p>"
    metric_names = (
        "num_problems",
        "total_generations",
        "avg_at_n",
        "pass_at_n",
        "majority_vote",
        "format_rate",
        "avg_completion_tokens",
    )
    header = "".join(f"<th>{html.escape(metric)}</th>" for metric in metric_names)
    body = []
    for condition, metrics in sorted(conditions.items()):
        metrics = metrics if isinstance(metrics, dict) else {}
        cells = []
        for metric in metric_names:
            value = metrics.get(metric, "")
            if isinstance(value, float):
                rendered = f"{value:.6f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append(f"<tr><th>{html.escape(str(condition))}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>condition</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html_report(
    cases: list[dict[str, Any]],
    rollout_summary: dict[str, Any],
    spikes: list[dict[str, Any]],
) -> str:
    summary = _report_summary(cases, rollout_summary, spikes)
    serializable_cases = []
    for case in cases:
        serializable_cases.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"reference_top_positions", "skeleton_top_positions"}
            }
        )
    payload = json.dumps(
        {"summary": summary, "cases": serializable_cases, "spikes": spikes},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    performance_table = _metric_table(rollout_summary)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher Base KL Contrast Visualization</title>
<style>
:root{{--reference:#dc2626;--skeleton:#2563eb;--border:#d7dce3;--muted:#596273}}
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1f2937}}main{{max-width:1500px;margin:auto;padding:24px}}
.panel{{background:#fff;border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:18px;overflow:auto}}
.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}.card{{background:#fff;border:1px solid var(--border);border-radius:10px;padding:14px}}
.value{{font-size:1.55rem;font-weight:700}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border-bottom:1px solid var(--border);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}
canvas{{width:100%;height:300px;border:1px solid var(--border);border-radius:8px}}.legend{{display:flex;gap:18px;color:var(--muted);margin:8px 0}}
.swatch{{display:inline-block;width:12px;height:12px;margin-right:5px}}#heatmap{{line-height:1.85;font-family:ui-monospace,monospace;overflow-wrap:anywhere}}.tok{{padding:2px 1px;border-radius:2px;cursor:default}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}select{{max-width:100%;padding:6px}}.dist{{max-width:360px;white-space:pre-wrap;text-align:left}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Teacher Base KL Contrast Visualization</h1>
<p>Same fixed <code>teacher_base</code> rollout tokens; red compares the reference teacher with base, blue compares the semantic-skeleton teacher with base.</p>
<section class="panel"><h2>Performance and token length</h2>{performance_table}</section>
<section class="cards">
  <div class="card"><div>Cases</div><div class="value">{summary['num_cases']}</div></div>
  <div class="card"><div>Mean reference KL</div><div class="value">{summary['mean_reference_kl']:.6f}</div></div>
  <div class="card"><div>Mean skeleton KL</div><div class="value">{summary['mean_skeleton_kl']:.6f}</div></div>
</section>
<section class="panel"><h2>Per-case KL curves and token heatmap</h2><select id="caseSelect"></select>
<div class="legend"><span><i class="swatch" style="background:var(--reference)"></i>reference</span><span><i class="swatch" style="background:var(--skeleton)"></i>skeleton</span></div>
<canvas id="chart" width="1400" height="320"></canvas><h3 id="caseTitle"></h3><div id="heatmap"></div></section>
<section class="panel"><h2>Top KL token positions with top distributions</h2><table><thead><tr><th>case</th><th>pos</th><th>token</th><th>reference KL</th><th>skeleton KL</th><th>reference top</th><th>skeleton top</th></tr></thead><tbody id="spikeRows"></tbody></table></section>
<script id="kl-data" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('kl-data').textContent), cases=data.cases, spikes=data.spikes;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const visible=s=>String(s??'').replace(/\n/g,'⏎').replace(/\t/g,'⇥').replace(/ /g,'·');
const topText=rows=>(rows||[]).slice(0,8).map(r=>`${{visible(r.token)}}: ${{Number(r.prob||0).toPrecision(3)}}`).join('\n');
const select=document.getElementById('caseSelect');
cases.forEach((c,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{c.case_id}} · tokens=${{c.num_tokens}}`;select.appendChild(o)}});
function draw(c){{
  document.getElementById('caseTitle').textContent=`${{c.case_id}} · correct=${{c.correct}} · completion_tokens=${{c.completion_tokens}}`;
  const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,p=34;
  ctx.clearRect(0,0,W,H);ctx.strokeStyle='#d7dce3';ctx.strokeRect(p,10,W-p-10,H-p-10);
  const max=Math.max(...c.reference_kl,...c.skeleton_kl,1e-9);
  [[c.reference_kl,'#dc2626'],[c.skeleton_kl,'#2563eb']].forEach(([arr,color])=>{{ctx.beginPath();ctx.strokeStyle=color;arr.forEach((v,i)=>{{const x=p+(W-p-10)*(arr.length<=1?0:i/(arr.length-1));const y=H-p-(H-p-10)*Math.log1p(Math.max(0,v))/Math.log1p(max);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke()}});
  const heat=document.getElementById('heatmap');heat.innerHTML='';
  c.tokens.forEach((token,i)=>{{const s=document.createElement('span'),r=c.reference_kl[i]||0,b=c.skeleton_kl[i]||0,t=Math.log1p(Math.max(r,b))/Math.log1p(max);s.className='tok';s.textContent=visible(token);s.title=`position ${{i}} | reference=${{r.toFixed(6)}} | skeleton=${{b.toFixed(6)}}`;s.style.background=`rgba(${{b>=r?'37,99,235':'220,38,38'}},${{0.08+0.82*t}})`;heat.appendChild(s)}});
}}
select.addEventListener('change',()=>draw(cases[Number(select.value)]));if(cases.length)draw(cases[0]);
document.getElementById('spikeRows').innerHTML=spikes.slice(0,500).map(r=>`<tr><td>${{esc(r.case_id)}}</td><td>${{r.position}}</td><td>${{esc(visible(r.token))}}</td><td>${{Number(r.reference_kl).toFixed(6)}}</td><td>${{Number(r.skeleton_kl).toFixed(6)}}</td><td class="dist">${{esc(topText(r.reference_teacher_top_tokens))}}</td><td class="dist">${{esc(topText(r.skeleton_teacher_top_tokens))}}</td></tr>`).join('');
</script></main></body></html>"""


def write_report_outputs(
    cases: list[dict[str, Any]],
    rollout_summary: dict[str, Any],
    csv_file: str | Path,
    spikes_jsonl_file: str | Path,
    report_file: str | Path,
) -> None:
    spikes = build_spike_rows(cases)
    csv_path = Path(csv_file)
    jsonl_path = Path(spikes_jsonl_file)
    html_path = Path(report_file)
    for output_path in (csv_path, jsonl_path, html_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in spikes)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in spikes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    html_path.write_text(render_html_report(cases, rollout_summary, spikes), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render reference-vs-skeleton KL comparisons on teacher_base trajectories."
    )
    parser.add_argument("--logit-file", required=True)
    parser.add_argument("--rollout-file", required=True)
    parser.add_argument("--rollout-summary-file", required=True)
    parser.add_argument("--skeleton-file", required=True)
    parser.add_argument("--csv-file", required=True)
    parser.add_argument("--spikes-jsonl-file", required=True)
    parser.add_argument("--report-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logit_records = read_jsonl(args.logit_file)
    rollout_records = read_jsonl(args.rollout_file)
    skeletons = read_skeleton_file(args.skeleton_file)
    with Path(args.rollout_summary_file).open("r", encoding="utf-8") as handle:
        rollout_summary = json.load(handle)
    cases = build_teacher_base_cases(logit_records, rollout_records, skeletons)
    write_report_outputs(
        cases=cases,
        rollout_summary=rollout_summary,
        csv_file=args.csv_file,
        spikes_jsonl_file=args.spikes_jsonl_file,
        report_file=args.report_file,
    )


if __name__ == "__main__":
    main()
