# Semantic Skeleton Prompt and GPU-Control Design

## Goal

Make the skeleton-conditioned teacher prompt explicitly describe its privileged context as a style-neutral semantic skeleton instead of presenting the JSON as a reference solution. Keep the reference-conditioned teacher prompt unchanged.

Also restore explicit GPU selection for skeleton OPSD training and synchronize the complete 2026-07-17 teacher KL-spike continuation workflow so training, KL comparison, and teacher continuation experiments can all target user-selected GPUs.

## Prompt Contract

The skeleton-conditioned prompt has this order and wording:

```text
Problem: {problem}
Below is a style-neutral semantic skeleton extracted from a reference solution.
=== Semantic Skeleton Begin ===
{skeleton_json}
=== Semantic Skeleton End ===
Interpret the fields as follows:
"key_objects" records potentially important mathematical objects and constraints.
"subgoals" records possible mathematical objectives.
"critical_intermediates" records potentially useful mathematical checkpoints. They are not mandatory generated sentences and do not imply that the reference path is the only valid path.
"theorem_tags" records optional and non-exclusive methods. Do not force a listed theorem when another valid approach is more natural.
"check" records validity conditions or possible failure modes. Apply a check only when it is relevant to the reasoning being used.
After reading the reference solution above, make sure you truly understand the reasoning. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and don't be afraid to backtrack or reconsider if something doesn't work out:
Please reason step by step, and put your final answer within \boxed{}.
```

The contract deliberately preserves the requested phrase `After reading the reference solution above`. The field name remains `critical_intermediates`, because that is the normalized key emitted in the skeleton JSON; the legacy singular key is accepted only as input.

## Data and Behavior

- Normalize the input skeleton using the existing normalization function.
- Continue omitting `final_answer` from the JSON block.
- Continue serializing `checks` as the prompt-facing key `check`.
- Do not add a separate `Final answer: {ground_truth}` line to the skeleton-conditioned prompt.
- Leave the reference-conditioned prompt and its transition text unchanged.
- Keep the skeleton builder's existing `ground_truth` parameter where it is part of a shared evaluation interface, but do not render it into the skeleton prompt.

## Code Scope

- `data_collator.py`: add skeleton-specific prompt copy and update the training prompt builder.
- `eval/quick_opsd_common.py`: mirror the same skeleton prompt for quick evaluations.
- `tests/test_opsd_skeleton_training.py`: assert the new training prompt boundaries, copy, field names, answer omission, and unchanged reference behavior.
- `tests/test_quick_opsd_common.py`: assert the mirrored quick-evaluation prompt.
- `tests/test_semantic_skeleton_scripts.py`: update integration-level skeleton prompt assertions.
- `docs/opsd_skeleton_training_zh.md` and `docs/semantic_skeleton_ablation.md`: document the new prompt contract.

## GPU-Control Contract

### OPSD training

Both `scripts/run_opsd_1b.sh` and `scripts/run_opsd_1b_skeleton.sh` expose the same environment variables:

- `TRAIN_GPU_IDS`: comma-separated physical GPU IDs passed through `CUDA_VISIBLE_DEVICES`, default `0,1,2,3`.
- `NUM_PROCESSES`: Accelerate worker count, default `4`; callers set it to the number of selected GPUs.
- `MAIN_PROCESS_PORT`: Accelerate rendezvous port, default `12949`.

Example:

```bash
TRAIN_GPU_IDS=4,5 NUM_PROCESSES=2 MAIN_PROCESS_PORT=12950 \
bash scripts/run_opsd_1b_skeleton.sh
```

The skeleton training script regains the configuration that was originally introduced in commit `c29e003` and accidentally overwritten in `d5cf0c4`. Model paths, skeleton paths, hyperparameters, run names, and W&B settings remain otherwise unchanged.

### KL comparison and teacher continuation

Synchronize commits `1701187`, `48c73b9`, `cdedd22`, and `0781e74` from `codex/teacher-kl-spike-continuations` into `main`. Together they provide:

- atomic ordered merging of rollout, KL, and continuation JSONL shards;
- generation of teacher continuations at global KL spikes;
- `--gpu-ids`/`--gpus` control for KL comparison and continuation workers;
- propagation of the selected GPU list from `run_student_teacher_category_kl.sh` into `run_teacher_spike_continuations.sh`;
- focused unit tests and experiment documentation.

KL shell runners accept a space-separated GPU list because they split the value into worker IDs with `read -r -a`, for example:

```bash
bash scripts/run_student_teacher_category_kl.sh \
  --gpu-ids "4 5" \
  --skeleton-file /path/to/skeletons.jsonl
```

The integrated KL runner performs student rollout, reference/skeleton teacher KL comparison, and teacher continuation generation. Passing `--skip-teacher-continuations` stops after the KL comparison; `scripts/run_teacher_spike_continuations.sh` can later resume continuation generation from the completed output directory using the same or a different GPU list.

## Integration Scope

- Merge the four 2026-07-17 commits as a complete dependency chain rather than copying individual files.
- Restore GPU variables only in `scripts/run_opsd_1b_skeleton.sh`; the reference training runner already has them.
- Preserve the user's existing `.DS_Store` modification.
- Do not pull or merge the unrelated `origin/main` commit while completing this change.
- Provide verified terminal commands for reference training, skeleton training, full KL comparison, KL-only execution, and continuation-only resume.

## Testing

Use a red-green cycle. First update the skeleton prompt tests and run them against the old implementation to verify failures caused by the missing semantic-skeleton wording. The existing skeleton-runner test already reproduces the missing training GPU configuration and must turn green after the script is restored. Run the synchronized 2026-07-17 tests for JSONL merging, KL continuation generation, and GPU-aware shell orchestration. Finally run the full test suite to detect unrelated prompt-shape or script dependencies.
