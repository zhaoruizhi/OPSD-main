import json
from pathlib import Path
from typing import Any, Iterable


SKELETON_SUBSET_POLICIES = {"error", "filter"}


def normalize_semantic_skeleton(skeleton: Any) -> dict[str, Any]:
    if isinstance(skeleton, str):
        skeleton = json.loads(skeleton)
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


def serialize_semantic_skeleton(skeleton: Any) -> str:
    return json.dumps(
        normalize_semantic_skeleton(skeleton),
        ensure_ascii=False,
        sort_keys=True,
    )


def read_skeleton_training_file(path: str | Path) -> dict[int, dict[str, Any]]:
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
        normalized = normalize_semantic_skeleton(skeleton)
        ground_truth = record.get("ground_truth")
        if ground_truth in (None, ""):
            ground_truth = normalized.get("final_answer")

        skeletons[int(record["problem_id"])] = {
            "semantic_skeleton": normalized,
            "ground_truth": str(ground_truth) if ground_truth not in (None, "") else None,
        }
    return skeletons


def attach_skeletons_to_training_rows(
    rows: Iterable[dict[str, Any]],
    skeleton_file: str | Path,
    subset_policy: str = "error",
) -> list[dict[str, Any]]:
    if subset_policy not in SKELETON_SUBSET_POLICIES:
        raise ValueError(
            f"skeleton_subset_policy must be one of {sorted(SKELETON_SUBSET_POLICIES)}, got {subset_policy!r}"
        )

    skeletons = read_skeleton_training_file(skeleton_file)
    prepared: list[dict[str, Any]] = []
    missing: list[int] = []

    for problem_id, row in enumerate(rows):
        skeleton_payload = skeletons.get(problem_id)
        if skeleton_payload is None:
            missing.append(problem_id)
            continue

        enriched = dict(row)
        enriched["semantic_skeleton"] = skeleton_payload["semantic_skeleton"]
        ground_truth = _ground_truth_for_row(enriched, skeleton_payload)
        if ground_truth not in (None, ""):
            enriched["ground_truth"] = str(ground_truth)
        prepared.append(enriched)

    if missing and subset_policy == "error":
        preview = ", ".join(str(problem_id) for problem_id in missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
        raise ValueError(f"missing skeletons for problem_id(s): {preview}{suffix}")

    if not prepared:
        raise ValueError("skeleton join produced an empty training dataset")

    return prepared


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_skeleton_training_rows(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ground_truth_for_row(row: dict[str, Any], skeleton_payload: dict[str, Any]) -> Any:
    for key in ("ground_truth", "Answer", "answer", "final_answer"):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return skeleton_payload.get("ground_truth")


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
