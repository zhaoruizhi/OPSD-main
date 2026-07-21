#!/usr/bin/env python3
"""Write an auditable manifest for semantic-skeleton comparison runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def build_experiment_config(
    *,
    effective_config: dict[str, Any],
    sample_indices_file: str | Path,
    skeleton_file: str | Path,
    git_commit: str,
    git_dirty: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "effective_config": effective_config,
        "input_files": {
            "sample_indices": _input_file_record(sample_indices_file),
            "skeletons": _input_file_record(skeleton_file),
        },
        "git": {"commit": git_commit, "dirty": bool(git_dirty)},
    }


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--experiment-profile", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sample-indices-file", required=True)
    parser.add_argument("--skeleton-file", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--val-n", type=int, required=True)
    parser.add_argument("--student-tm", choices=["off", "on"], required=True)
    parser.add_argument("--student-max-new-tokens", type=int, required=True)
    parser.add_argument("--teacher-max-new-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--trajectory-sample-index", type=int, required=True)
    parser.add_argument("--probe-tokens", type=int, required=True)
    parser.add_argument("--target-token-source", required=True)
    parser.add_argument("--hf-device-map", required=True)
    parser.add_argument("--teacher-continuations", choices=["enabled", "skipped"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    git_commit, git_dirty = _git_state(repo_root)
    effective_config = {
        "experiment_profile": args.experiment_profile,
        "base_model": args.base_model,
        "checkpoint_dir": args.checkpoint_dir or None,
        "dataset": args.dataset,
        "split": args.split,
        "sample_size": args.sample_size,
        "val_n": args.val_n,
        "student_tm": args.student_tm,
        "student_max_new_tokens": args.student_max_new_tokens,
        "teacher_max_new_tokens": args.teacher_max_new_tokens,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "gpu_ids": [int(value) for value in args.gpu_ids.split()],
        "trajectory_sample_index": args.trajectory_sample_index,
        "probe_tokens": args.probe_tokens,
        "target_token_source": args.target_token_source,
        "hf_device_map": args.hf_device_map,
        "teacher_continuations": args.teacher_continuations,
    }
    payload = build_experiment_config(
        effective_config=effective_config,
        sample_indices_file=args.sample_indices_file,
        skeleton_file=args.skeleton_file,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
