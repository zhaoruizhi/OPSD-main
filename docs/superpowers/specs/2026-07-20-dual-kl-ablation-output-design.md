# Dual-KL Ablation Output Compatibility Design

**Date:** 2026-07-20

## Goal

Extend `scripts/run_student_teacher_category_kl.sh` so one invocation produces the
same core evaluation artifacts as the earlier semantic-skeleton ablation run,
while retaining the newer student-trajectory KL probe and teacher spike
continuations.

The teacher prompts must continue to use the current prompt builders. In
particular, the semantic-skeleton teacher must use the current style-neutral
semantic-skeleton prompt and must not revert to the old reference-solution
wording.

## Why the Current Output Is Incomplete

The current runner generates only the `student` rollout. It then evaluates
`teacher_reference` and `teacher_skeleton` on the fixed student completion and
explicitly skips rollout entropy. Consequently:

- performance and completion-token statistics exist only for `student`;
- there is no standalone `teacher_base`, `teacher_reference`, or
  `teacher_skeleton` rollout summary;
- the old teacher-base-relative KL comparison is absent;
- the old KL visualization artifacts cannot be generated from the new output.

This is an experiment-wiring difference, not a summary parser failure.

## Chosen Design

The runner will execute three logically separate phases after sampling the
shared problem set.

### 1. Four-condition rollout evaluation

Generate rollouts for these conditions using the current prompt code:

- `student`
- `teacher_base`
- `teacher_reference`
- `teacher_skeleton`

All conditions use the same sampled problem IDs. Each rollout record preserves
the prompt and completion token IDs already emitted by the rollout evaluator.
The merged artifacts use the earlier ablation-compatible names:

- `rollouts.jsonl`
- `rollout_summary.json`

The summary reports performance and completion-token statistics for all four
conditions. Per-shard files remain available for debugging and resumption.

### 2. Two KL probes

Run both probes because they answer different questions.

#### Teacher-base trajectory KL (ablation compatibility)

Use `teacher_base` completions as the fixed target trajectory and compare:

- `teacher_reference` against `teacher_base`
- `teacher_skeleton` against `teacher_base`

Also compute rollout entropy for all four conditions. Emit the earlier
ablation-compatible artifacts:

- `logit_probe.jsonl`
- `logit_summary.json`
- shard-level probe and summary files
- teacher-base KL report, top-spike CSV, and spike JSONL visualizations

This probe is used for the old-style reference-versus-skeleton KL comparison.

#### Student trajectory KL (continuation source)

Use the student completion IDs as the fixed target trajectory and compare:

- `teacher_reference` against `student`
- `teacher_skeleton` against `student`

Emit the existing `student_teacher_category_kl*` artifacts. These records remain
the source for selecting teacher KL spikes and generating continuations, so the
meaning of the existing continuation experiment does not change.

The two probes must never overwrite each other. Their filenames, summaries,
and console phase labels must clearly identify the target trajectory and
baseline.

### 3. Teacher spike continuations

Continue to select high-KL positions from the student-trajectory probe. Preserve:

- `teacher_spike_continuations.jsonl`
- `teacher_spike_continuation_summary.json`
- shard continuation files
- `visualizations/teacher_spike_continuations.html`

Existing continuation controls, including top-N and maximum new-token limits,
remain available.

## GPU Assignment

`--gpu-ids` remains the single user-facing device selector. Shards are assigned
only to those IDs for rollout generation, both KL probes, and continuations.
The runner must preserve the existing `CUDA_VISIBLE_DEVICES`-based isolation so
the experiment does not use GPUs outside the supplied list.

## Compatibility and Naming

The final output directory is a superset of the earlier ablation directory:

- old-compatible four-condition rollout, performance, token-length, KL,
  entropy, and visualization artifacts;
- new student-trajectory category KL artifacts;
- new teacher continuation artifacts.

Compatibility means matching the old artifact schema and experiment conditions,
not forcing the same sample count. `--sample-size` and the rollout samples per
problem continue to determine record counts.

## Failure Handling

- A phase must fail if any shard process fails.
- Merges must reject malformed JSONL and should be atomic.
- All four rollout conditions must be present before either KL phase begins.
- The continuation phase must consume only the student-trajectory KL file.
- Empty selections must produce valid empty outputs and summaries rather than
  silently reading a file from the other KL probe.

## Verification

Add or update tests to verify:

1. the runner launches all four current-prompt rollout conditions;
2. the teacher-base probe uses `teacher_base` as trajectory and baseline and
   does not skip rollout entropy;
3. the student probe still uses the student trajectory and emits its existing
   filenames;
4. continuation reads the student-trajectory KL output;
5. `--gpu-ids` is applied across every GPU phase;
6. final expected filenames include performance/token-length summaries, both
   KL families, old-compatible visualizations, and continuation artifacts;
7. a small fixture run produces summaries with all four conditions and both KL
   contrasts without requiring model downloads.

