# Legacy Skeleton Training Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `run_opsd_1b_skeleton.sh` to train with the exact validated `legacy-20260629` skeleton teacher prompt while keeping the current style-neutral prompt as the default and recording a reproducible training manifest.

**Architecture:** Reuse the profile-aware prompt builder in `eval/quick_opsd_common.py` from the training collator so rollout/KL/training cannot drift. Propagate one validated `teacher_prompt_profile` value from the shell runner through `opsd_train.py` and `OPSDTrainer` to `SelfDistillationDataCollator`. Write a rank-zero manifest from the parsed effective training/model/script arguments, Git state, command line, runtime environment, and skeleton SHA-256.

**Tech Stack:** Python 3, PyTorch/Transformers/TRL, Bash, `unittest`/pytest, JSON.

## Global Constraints

- Preserve `current-style-neutral` as the default profile.
- `legacy-20260629` must use the same `build_semantic_skeleton_user_message(...)` implementation already used by rollout, both KL probes, and teacher continuations.
- The legacy prompt must include `Final answer:` after the semantic skeleton block and serialize the field as `checks`.
- The 128-row KL reproduction skeleton file is not valid for full training; the documented command must use the full-train skeleton file.
- Keep OPSD training `max_completion_length=1024`; the KL reproduction limit of 16384 is not a training default.
- Do not add a new dependency.

---

### Task 1: Profile-aware skeleton training prompt

**Files:**
- Modify: `tests/test_opsd_skeleton_training.py`
- Modify: `data_collator.py`

**Interfaces:**
- Consumes: `build_semantic_skeleton_user_message(problem, skeleton, ground_truth, teacher_prompt_profile)` and `DEFAULT_TEACHER_PROMPT_PROFILE` from `eval.quick_opsd_common`.
- Produces: `SelfDistillationDataCollator(..., teacher_prompt_profile: str = DEFAULT_TEACHER_PROMPT_PROFILE)`.

- [ ] **Step 1: Add failing tests for exact legacy prompt output and default current behavior**

  Add a test that constructs the collator with `teacher_prompt_profile="legacy-20260629"`, feeds a skeleton with answer `4`, and asserts that the teacher message contains the legacy semantic-skeleton wrapper, plural `checks`, and `Final answer: 4` after `=== Semantic Skeleton End ===`. Preserve the existing current-profile assertions.

- [ ] **Step 2: Run the focused tests and verify the legacy test fails because the collator lacks the argument**

  Run: `python3 -m pytest tests/test_opsd_skeleton_training.py -q`
  Expected: FAIL with an unexpected `teacher_prompt_profile` argument.

- [ ] **Step 3: Delegate skeleton prompt construction to the shared profile-aware builder**

  Import the shared builder/default, validate the selected profile at collator initialization, normalize serialized training skeletons, and call the shared builder with `feature.get("ground_truth")`.

- [ ] **Step 4: Run the focused tests**

  Run: `python3 -m pytest tests/test_opsd_skeleton_training.py -q`
  Expected: PASS.

### Task 2: Propagate profile and configurable run name

**Files:**
- Modify: `tests/test_opsd_skeleton_training.py`
- Modify: `opsd_train.py`
- Modify: `opsd_trainer.py`
- Modify: `scripts/run_opsd_1b_skeleton.sh`

**Interfaces:**
- Consumes: `teacher_prompt_profile` from the CLI and `TEACHER_PROMPT_PROFILE`/`RUN_CONFIG` from the runner environment.
- Produces: `OPSDTrainer(..., teacher_prompt_profile: str)` and the CLI option `--teacher_prompt_profile {current-style-neutral,legacy-20260629}`.

- [ ] **Step 1: Add failing source/runner tests for profile and run-name propagation**

  Assert the runner defines `TEACHER_PROMPT_PROFILE` and `RUN_CONFIG`, passes quoted values to `opsd_train.py`, and that `opsd_train.py` passes the parsed profile to `OPSDTrainer`.

- [ ] **Step 2: Run the focused tests and verify they fail on missing propagation**

  Run: `python3 -m pytest tests/test_opsd_skeleton_training.py -q`
  Expected: FAIL on missing environment variables/CLI propagation.

- [ ] **Step 3: Implement the CLI field and propagation**

  Add the dataclass field with the two accepted choices, include it in W&B config, pass it to `OPSDTrainer`, then to `SelfDistillationDataCollator`, and retain it on the trainer for diagnostics.

- [ ] **Step 4: Make the runner values overrideable without changing existing defaults**

  Define `TEACHER_PROMPT_PROFILE`, `RUN_CONFIG`, `MODEL_NAME_OR_PATH`, and `OUTPUT_DIR` environment defaults and quote each use in the launch command.

- [ ] **Step 5: Run the focused tests**

  Run: `python3 -m pytest tests/test_opsd_skeleton_training.py -q`
  Expected: PASS.

### Task 3: Reproducible training manifest and runbook

**Files:**
- Create: `training_experiment_manifest.py`
- Create: `tests/test_training_experiment_manifest.py`
- Modify: `opsd_train.py`
- Modify: `docs/opsd_skeleton_training_zh.md`

**Interfaces:**
- Produces: `write_training_experiment_manifest(output_file, *, script_args, training_args, model_args, skeleton_file, repo_root, argv, environ)`.
- Output: `<training output>/<run_config>/experiment_config.json`.

- [ ] **Step 1: Add failing manifest tests**

  Use temporary skeleton and repository directories. Assert schema version, serialized effective config groups, command line, selected runtime variables, Git commit/dirty state, and skeleton SHA-256.

- [ ] **Step 2: Run the manifest test and verify it fails because the helper is missing**

  Run: `python3 -m pytest tests/test_training_experiment_manifest.py -q`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dependency-free manifest helper**

  Serialize dataclasses/`to_dict()` configs through a JSON-safe converter, record UTC time, command, selected distributed/CUDA environment, Git state, and resolved skeleton path/hash. Create parent directories before writing.

- [ ] **Step 4: Write the manifest once on rank zero before W&B/model initialization**

  Call the helper after `training_args.output_dir` has been resolved with `run_config`; include `teacher_prompt_profile` in the printed configuration.

- [ ] **Step 5: Document the exact legacy training preflight and launch command**

  Document that the reproduction directory contains only 128 skeleton rows, use `/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl`, set `TEACHER_PROMPT_PROFILE=legacy-20260629`, and keep `max_completion_length=1024`.

- [ ] **Step 6: Run focused and full verification**

  Run:

  ```bash
  python3 -m pytest tests/test_training_experiment_manifest.py tests/test_opsd_skeleton_training.py -q
  python3 -m pytest -q
  bash -n scripts/run_opsd_1b_skeleton.sh
  git diff --check
  ```

  Expected: all tests pass, shell syntax exits 0, and `git diff --check` produces no output.

