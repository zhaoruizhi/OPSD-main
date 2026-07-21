# Legacy 2026-06-29 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `legacy-20260629` experiment profile that reproduces the archived semantic-skeleton prompt and rollout settings while preserving the current style-neutral prompt as a separate profile.

**Architecture:** Prompt construction remains centralized in `eval/quick_opsd_common.py` and gains a validated profile argument. The selected profile is propagated through rollout, KL prompt reconstruction, and spike continuation workers. The dual-KL shell runner exposes the profile and `--val-n`, applies artifact-backed legacy defaults, and records the effective configuration for auditability.

**Tech Stack:** Python 3, Bash, `unittest`, vLLM/HuggingFace entrypoints already present in the repository.

## Global Constraints

- Work directly on the existing `main` worktree; do not create a feature worktree.
- Preserve `current-style-neutral` behavior when no legacy profile is selected.
- Reproduce the prompt captured in root commit `7f414c68f0f55a47fb86e7aa8badc3d477112188` exactly, including the standalone `Final answer:` line and plural `checks` JSON field.
- Legacy rollout settings are `sample_size=128`, `val_n=4`, student thinking off, student/teacher max-new-tokens `16384`, temperature `1.1`, top-p `0.95`, top-k `20`, max-model-len `20000`, seed `0`, and trajectory sample index `0`.
- Legacy KL uses `full_generation` / `target_tail_text` text re-tokenization even when rollout token IDs are available; the current profile continues to use original rollout token IDs.
- The legacy profile must require explicit archived sample-manifest and skeleton files; GPU IDs remain user-selectable, with four GPUs recommended for the closest stochastic reproduction.
- Preserve teacher continuation outputs and use the same selected prompt profile during continuation reconstruction.

---

### Task 1: Exact legacy prompt profile

**Files:**
- Modify: `eval/quick_opsd_common.py`
- Test: `tests/test_quick_opsd_common.py`
- Test: `tests/test_semantic_skeleton_scripts.py`

**Interfaces:**
- Produces: `TEACHER_PROMPT_PROFILES`, `DEFAULT_TEACHER_PROMPT_PROFILE`, and profile-aware `build_reference_user_message(...)` / `build_semantic_skeleton_user_message(...)`.
- Preserves: existing callers that omit the profile continue using `current-style-neutral`.

- [ ] **Step 1: Write failing tests for exact legacy reference and skeleton prompt strings**

```python
legacy_reference = build_reference_user_message(
    "Compute 2+2.",
    "A reference solution says 2+2=4.",
    ground_truth="4",
    teacher_prompt_profile="legacy-20260629",
)
legacy_skeleton = build_semantic_skeleton_user_message(
    "Compute 2+2.",
    skeleton,
    ground_truth="4",
    teacher_prompt_profile="legacy-20260629",
)
```

- [ ] **Step 2: Run the focused prompt tests and confirm they fail because the profile argument is missing**

Run: `python -m unittest tests.test_quick_opsd_common tests.test_semantic_skeleton_scripts`

- [ ] **Step 3: Implement validated current and legacy prompt branches using the exact `7f414c68` text**

- [ ] **Step 4: Re-run the focused tests and confirm both current and legacy prompt behavior pass**

Run: `python -m unittest tests.test_quick_opsd_common tests.test_semantic_skeleton_scripts`

### Task 2: Propagate the profile through rollout, both KL probes, and continuations

**Files:**
- Modify: `eval/quick_rollout_openthoughts.py`
- Modify: `eval/quick_logit_probe.py`
- Modify: `eval/quick_teacher_spike_continuation.py`
- Modify: `scripts/run_teacher_spike_continuations.sh`
- Test: `tests/test_quick_opsd_scripts.py`
- Test: `tests/test_quick_logit_probe.py`
- Test: `tests/test_teacher_spike_continuation.py`
- Test: `tests/test_quick_opsd_run_script.py`

**Interfaces:**
- Consumes: the prompt-profile constants and builder arguments from Task 1.
- Produces: `--teacher-prompt-profile {current-style-neutral,legacy-20260629}` on every prompt-reconstructing entrypoint.
- Produces: `--target-token-source {auto,target_tail_text}` on the KL probe, with legacy runs selecting `target_tail_text`.

- [ ] **Step 1: Add failing propagation tests for rollout, KL reconstruction, legacy target-text tokenization, and continuation worker arguments**

- [ ] **Step 2: Run focused tests and confirm failures identify missing CLI/profile propagation**

Run: `python -m unittest tests.test_quick_opsd_scripts tests.test_quick_logit_probe tests.test_teacher_spike_continuation tests.test_quick_opsd_run_script`

- [ ] **Step 3: Add the prompt-profile CLI option and pass it into every reference/skeleton prompt reconstruction call**

- [ ] **Step 4: Add selectable target-token sourcing to the KL probe and make continuation positions use the same token sequence selected by KL**

- [ ] **Step 5: Record `teacher_prompt_profile` in new rollout records**

- [ ] **Step 6: Re-run focused tests and confirm they pass**

Run: `python -m unittest tests.test_quick_opsd_scripts tests.test_quick_logit_probe tests.test_teacher_spike_continuation tests.test_quick_opsd_run_script`

### Task 3: Legacy runner settings and experiment manifest

**Files:**
- Modify: `scripts/run_student_teacher_category_kl.sh`
- Create: `eval/write_experiment_config.py`
- Test: `tests/test_experiment_config.py`
- Test: `tests/test_quick_opsd_run_script.py`

**Interfaces:**
- Consumes: `--teacher-prompt-profile` from Tasks 1-2.
- Produces: runner options `--experiment-profile` and `--val-n`, plus `$OUT/experiment_config.json` containing effective paths, hashes, sampling settings, token limits, GPU IDs, and Git commit.

- [ ] **Step 1: Write failing tests for `--val-n`, legacy defaults, required archived inputs, and manifest fields**

- [ ] **Step 2: Run focused tests and confirm they fail for the missing runner/config behavior**

Run: `python -m unittest tests.test_experiment_config tests.test_quick_opsd_run_script`

- [ ] **Step 3: Implement runner parsing and profile defaults**

```text
legacy-20260629:
  sample_size=128
  val_n=4
  student_tm=off
  student_max_new_tokens=16384
  teacher_max_new_tokens=16384
  max_model_len=20000
  temperature=1.1
  top_p=0.95
  top_k=20
  seed=0
  trajectory_sample_index=0
```

- [ ] **Step 4: Implement deterministic experiment-config writing with SHA-256 hashes for the copied manifest and skeleton file**

- [ ] **Step 5: Pass the selected prompt profile through rollout, teacher-base KL, student KL, and continuation phases**

- [ ] **Step 6: Pass `target_tail_text` tokenization to both KL phases for `legacy-20260629`; leave current runs on `auto`**

- [ ] **Step 7: Re-run focused tests and Bash syntax checks**

Run: `python -m unittest tests.test_experiment_config tests.test_quick_opsd_run_script`

Run: `bash -n scripts/run_student_teacher_category_kl.sh scripts/run_teacher_spike_continuations.sh`

### Task 4: Chinese runbook and full verification

**Files:**
- Modify: `docs/experiment_runbook_zh.md`
- Modify: `docs/student_teacher_category_kl_zh.md`

**Interfaces:**
- Documents: one exact legacy reproduction command and one current-style-neutral command, including which archived files to reuse and how `avg_at_n`, `pass_at_n`, token length, both KL outputs, and continuations are read.

- [ ] **Step 1: Update both documents with explicit profile names and complete server commands**

- [ ] **Step 2: Verify no old statement still claims the dual-KL runner is fixed at `VAL_N=1`**

Run: `rg -n "VAL_N=1|默认.*n=1|teacher prompt 始终来自当前代码" docs scripts tests`

- [ ] **Step 3: Run the complete relevant test suite**

Run: `python -m unittest tests.test_quick_opsd_common tests.test_semantic_skeleton_scripts tests.test_quick_opsd_scripts tests.test_quick_logit_probe tests.test_teacher_spike_continuation tests.test_experiment_config tests.test_quick_opsd_run_script`

- [ ] **Step 4: Run syntax/compile verification and inspect the final diff**

Run: `python -m py_compile eval/quick_opsd_common.py eval/quick_rollout_openthoughts.py eval/quick_logit_probe.py eval/quick_teacher_spike_continuation.py eval/write_experiment_config.py`

Run: `bash -n scripts/run_student_teacher_category_kl.sh scripts/run_teacher_spike_continuations.sh`

Run: `git diff --check`
