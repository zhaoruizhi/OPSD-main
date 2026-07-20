# Semantic Skeleton Prompt and GPU-Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the skeleton-conditioned teacher prompt's reference-solution framing, restore selected-GPU training, synchronize the 2026-07-17 KL-spike continuation workflow, and document reproducible commands for every experiment.

**Architecture:** Keep the reference-conditioned prompt untouched. Define skeleton-specific prompt copy in both prompt-building modules. Merge the complete four-commit KL continuation dependency chain, restore the skeleton training runner's environment-based GPU controls, and expose training, integrated KL, KL-only, and continuation-only workflows through documented shell commands.

**Tech Stack:** Python 3.10, `unittest`, Transformers-compatible prompt builders, Bash runners, Markdown documentation.

## Global Constraints

- The skeleton prompt must match `docs/superpowers/specs/2026-07-20-semantic-skeleton-prompt-design.md`.
- The prompt-facing normalized field remains `critical_intermediates`.
- Skeleton JSON omits `final_answer` and maps `checks` to `check`.
- Skeleton prompts do not render a separate `Final answer:` line.
- Reference prompts remain behaviorally unchanged.
- Merge commits `1701187`, `48c73b9`, `cdedd22`, and `0781e74` from `codex/teacher-kl-spike-continuations` as one dependency chain.
- Training GPU IDs are comma-separated in `TRAIN_GPU_IDS`; KL runner GPU IDs are space-separated in `--gpu-ids`.
- Document reference training, skeleton training, integrated KL plus continuation, KL-only, and continuation-only resume commands.
- Do not modify the user's unrelated `.DS_Store` change.

---

### Task 0: Synchronize the 2026-07-17 KL continuation workflow

**Files:**
- Merge: `codex/teacher-kl-spike-continuations`
- Create: `eval/quick_jsonl_merge.py`
- Create: `eval/quick_teacher_spike_continuation.py`
- Create: `scripts/run_teacher_spike_continuations.sh`
- Modify: `scripts/run_student_teacher_category_kl.sh`
- Create: `tests/test_quick_jsonl_merge.py`
- Create: `tests/test_teacher_spike_continuation.py`
- Modify: `tests/test_quick_opsd_run_script.py`
- Modify/Create: KL continuation documentation from the branch

**Interfaces:**
- Consumes: a completed or in-progress student/teacher category-KL output directory and a space-separated `--gpu-ids` value.
- Produces: ordered JSONL aggregates, teacher continuations at global KL spikes, summaries, and an HTML comparison report.

- [ ] **Step 1: Merge the complete branch**

Run:

```bash
git merge --no-ff codex/teacher-kl-spike-continuations
```

Expected: the four commits after common ancestor `84ce1a9` merge without modifying `.DS_Store`.

- [ ] **Step 2: Run the synchronized focused tests**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_quick_jsonl_merge.py' -q
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_teacher_spike_continuation.py' -q
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_quick_opsd_run_script.py' -q
```

Expected: all synchronized tests pass.

---

### Task 0.5: Restore selected-GPU skeleton training

**Files:**
- Modify: `scripts/run_opsd_1b_skeleton.sh:1-12`
- Test: `tests/test_opsd_skeleton_training.py:253-267`

**Interfaces:**
- Consumes: comma-separated `TRAIN_GPU_IDS`, integer `NUM_PROCESSES`, and integer `MAIN_PROCESS_PORT` environment variables.
- Produces: an Accelerate launch restricted by `CUDA_VISIBLE_DEVICES` to the selected physical GPUs.

- [ ] **Step 1: Confirm the existing regression test is RED**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_opsd_skeleton_training.py' -q
```

Expected: only `test_skeleton_run_script_uses_skeleton_mode_and_distinct_run_config` fails before the prompt tests are changed.

- [ ] **Step 2: Restore the same launcher controls used by reference training**

Prepend and wire:

```bash
#!/usr/bin/env bash
set -euo pipefail

SKELETON_FILE="${SKELETON_FILE:-/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps 2 \
    --main_process_port "$MAIN_PROCESS_PORT" \
```

Leave the remaining skeleton training arguments unchanged.

- [ ] **Step 3: Run the existing runner test and verify GREEN for GPU selection**

Run the test file again. Expected: the runner assertion passes; newly introduced prompt assertions may still fail until Task 3.

---

### Task 1: Specify the new training prompt behavior

**Files:**
- Modify: `tests/test_opsd_skeleton_training.py:159-215`
- Test: `tests/test_opsd_skeleton_training.py`

**Interfaces:**
- Consumes: `SelfDistillationDataCollator(..., teacher_context_mode="skeleton", reason_first=False)`
- Produces: regression assertions for the new training prompt contract.

- [ ] **Step 1: Update the skeleton prompt test before production code**

Change the skeleton block extraction and assertions to require:

```python
skeleton_block = teacher_content.split("=== Semantic Skeleton Begin ===\n", 1)[1].split(
    "\n=== Semantic Skeleton End ===", 1
)[0]
self.assertNotIn("Final answer:", teacher_content)
self.assertIn(
    "Problem: Compute 2+2.\nBelow is a style-neutral semantic skeleton extracted from a reference solution.\n"
    "=== Semantic Skeleton Begin ===",
    teacher_content,
)
self.assertNotIn("=== Reference Solution Begin ===", teacher_content)
self.assertNotIn("Here is a reference solution to this problem:", teacher_content)
self.assertIn('"critical_intermediates"', skeleton_block)
self.assertIn('"check"', skeleton_block)
self.assertNotIn('"checks"', skeleton_block)
self.assertNotIn("final_answer", skeleton_block)
self.assertIn(
    "=== Semantic Skeleton End ===\nInterpret the fields as follows:\n"
    '"key_objects" records potentially important mathematical objects and constraints.\n',
    teacher_content,
)
self.assertIn(
    "After reading the reference solution above, make sure you truly understand the reasoning. Now, using your own words",
    teacher_content,
)
self.assertNotIn("reasoning behind each step — do not copy or paraphrase it", teacher_content)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_opsd_skeleton_training.py' -q
```

Expected: the skeleton-mode test fails because the old implementation still emits `Reference Solution` boundaries and `Final answer: 4`.

---

### Task 2: Specify the mirrored quick-evaluation prompt

**Files:**
- Modify: `tests/test_quick_opsd_common.py:79-122`
- Modify: `tests/test_semantic_skeleton_scripts.py:755-783`
- Test: `tests/test_quick_opsd_common.py`
- Test: `tests/test_semantic_skeleton_scripts.py`

**Interfaces:**
- Consumes: `build_semantic_skeleton_user_message(problem, skeleton, ground_truth=None) -> str`
- Produces: regression assertions proving quick evaluation uses the same prompt as training.

- [ ] **Step 1: Update quick prompt assertions**

Require the style-neutral heading, `Semantic Skeleton` boundaries, plural `critical_intermediates`, singular prompt-facing `check`, absence of `Final answer:`, and the approved shortened transition. Extract the block with:

```python
skeleton_block = prompt.split("=== Semantic Skeleton Begin ===\n", 1)[1].split(
    "\n=== Semantic Skeleton End ===", 1
)[0]
```

Replace the old negative semantic-skeleton assertions with:

```python
self.assertIn("Below is a style-neutral semantic skeleton extracted from a reference solution.", prompt)
self.assertIn("=== Semantic Skeleton Begin ===", prompt)
self.assertIn("=== Semantic Skeleton End ===", prompt)
self.assertNotIn("Reference Solution Begin", prompt)
self.assertNotIn("Final answer:", prompt)
```

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_quick_opsd_common.py' -q
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_semantic_skeleton_scripts.py' -q
```

Expected: skeleton-prompt tests fail on the old heading, boundaries, ground-truth line, and transition; reference-prompt tests remain valid.

---

### Task 3: Implement the training and evaluation prompts

**Files:**
- Modify: `data_collator.py:10-20,247-277`
- Modify: `eval/quick_opsd_common.py:22-32,344-376`
- Test: `tests/test_opsd_skeleton_training.py`
- Test: `tests/test_quick_opsd_common.py`
- Test: `tests/test_semantic_skeleton_scripts.py`

**Interfaces:**
- Consumes: normalized semantic skeleton dictionaries or serialized skeleton input accepted by existing normalization.
- Produces: skeleton-only prompt text following the approved contract; reference builder interfaces remain unchanged.

- [ ] **Step 1: Add skeleton-specific transition constants**

In both prompt modules, define:

```python
SEMANTIC_SKELETON_TRANSITION_PROMPT = (
    "After reading the reference solution above, make sure you truly understand the reasoning. "
    "Now, using your own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:"
)
```

Remove the leading `- ` from each field-guidance line while retaining `critical_intermediates`.

- [ ] **Step 2: Update the training skeleton builder**

Remove the skeleton builder's local ground-truth lookup and rendered `ground_truth_line`. Keep the answer-free JSON construction, then return:

```python
return (
    f"Problem: {problem}\n"
    "Below is a style-neutral semantic skeleton extracted from a reference solution.\n"
    f"=== Semantic Skeleton Begin ===\n{skeleton_json}\n=== Semantic Skeleton End ===\n"
    f"{SEMANTIC_SKELETON_FIELD_GUIDANCE}\n"
    f"{SEMANTIC_SKELETON_TRANSITION_PROMPT}\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)
```

- [ ] **Step 3: Update the quick-evaluation skeleton builder**

Keep the public `ground_truth` parameter for caller compatibility but do not render it. Remove the local `final_answer` computation and return the same prompt shape as training using `skeleton_json`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_opsd_skeleton_training.py' -q
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_quick_opsd_common.py' -q
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -p 'test_semantic_skeleton_scripts.py' -q
```

Expected: all focused tests pass.

---

### Task 4: Synchronize documentation and verify the repository

**Files:**
- Modify: `docs/opsd_skeleton_training_zh.md:63-90`
- Modify: `docs/semantic_skeleton_ablation.md:51-80`
- Modify: `docs/student_teacher_category_kl_zh.md`
- Modify: `docs/teacher_kl_spike_continuations_zh.md`
- Test: repository test suite

**Interfaces:**
- Consumes: the implemented prompt contract.
- Produces: user-facing documentation that matches emitted training and evaluation text.

- [ ] **Step 1: Update both prompt examples and behavior notes**

Replace `Final answer:`, the reference-solution heading/boundaries, bullet-prefixed field guidance, and the old transition with the approved prompt. State that neither the JSON block nor another line exposes `final_answer` in skeleton mode. Add complete, copyable command sections for:

```bash
# Reference OPSD training
TRAIN_GPU_IDS=4,5 NUM_PROCESSES=2 MAIN_PROCESS_PORT=12949 \
bash scripts/run_opsd_1b.sh

# Skeleton OPSD training
SKELETON_FILE=/path/to/skeletons.jsonl \
TRAIN_GPU_IDS=4,5 NUM_PROCESSES=2 MAIN_PROCESS_PORT=12950 \
bash scripts/run_opsd_1b_skeleton.sh

# Integrated rollout + KL comparison + teacher continuation
bash scripts/run_student_teacher_category_kl.sh \
  --base-model /path/to/base-model \
  --checkpoint-dir /path/to/checkpoint \
  --skeleton-file /path/to/skeletons.jsonl \
  --gpu-ids "4 5" \
  --out /path/to/output

# KL comparison only
bash scripts/run_student_teacher_category_kl.sh \
  --base-model /path/to/base-model \
  --checkpoint-dir /path/to/checkpoint \
  --skeleton-file /path/to/skeletons.jsonl \
  --gpu-ids "4 5" \
  --out /path/to/output \
  --skip-teacher-continuations

# Continuation-only resume
bash scripts/run_teacher_spike_continuations.sh \
  --base-model /path/to/base-model \
  --checkpoint-dir /path/to/checkpoint \
  --out /path/to/completed-kl-output \
  --gpu-ids "4 5" \
  --top-n 10 \
  --max-new-tokens 20
```

For each command, document prerequisites, GPU ID delimiter, output artifacts, and how integrated versus resumed continuation runs differ.

- [ ] **Step 2: Run formatting and stale-copy checks**

Run:

```bash
git diff --check
rg -n -F 'Here is a reference solution to this problem:' data_collator.py eval/quick_opsd_common.py tests docs
rg -n -F 'Semantic Skeleton Begin' data_collator.py eval/quick_opsd_common.py tests docs
```

Expected: no whitespace errors; remaining reference-solution matches belong only to reference-mode builders/tests/docs; semantic-skeleton matches cover training, quick evaluation, tests, and docs.

- [ ] **Step 3: Run the full test suite**

Run:

```bash
/Users/zhaoruizhi/miniconda3/envs/etapp/bin/python -m unittest discover -s tests -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Review final scope**

Run:

```bash
git status --short
git diff -- data_collator.py eval/quick_opsd_common.py tests/test_opsd_skeleton_training.py tests/test_quick_opsd_common.py tests/test_semantic_skeleton_scripts.py docs/opsd_skeleton_training_zh.md docs/semantic_skeleton_ablation.md
```

Expected: only planned files plus the plan document are changed; `.DS_Store` remains untouched as a pre-existing user modification.
