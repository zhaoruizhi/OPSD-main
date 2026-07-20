# Semantic Skeleton Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the skeleton-conditioned teacher prompt's reference-solution framing with the approved style-neutral semantic-skeleton prompt in training and quick evaluation.

**Architecture:** Keep the reference-conditioned prompt untouched. Define skeleton-specific field guidance and transition copy in each existing prompt-building module, then make both skeleton builders serialize the same answer-free normalized JSON inside `Semantic Skeleton` boundaries.

**Tech Stack:** Python 3, `unittest`/pytest, Transformers-compatible prompt builders, Markdown documentation.

## Global Constraints

- The skeleton prompt must match `docs/superpowers/specs/2026-07-20-semantic-skeleton-prompt-design.md`.
- The prompt-facing normalized field remains `critical_intermediates`.
- Skeleton JSON omits `final_answer` and maps `checks` to `check`.
- Skeleton prompts do not render a separate `Final answer:` line.
- Reference prompts remain behaviorally unchanged.
- Do not modify the user's unrelated `.DS_Store` change.

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
python -m pytest tests/test_opsd_skeleton_training.py -q
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
python -m pytest tests/test_quick_opsd_common.py tests/test_semantic_skeleton_scripts.py -q
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
python -m pytest tests/test_opsd_skeleton_training.py tests/test_quick_opsd_common.py tests/test_semantic_skeleton_scripts.py -q
```

Expected: all focused tests pass.

---

### Task 4: Synchronize documentation and verify the repository

**Files:**
- Modify: `docs/opsd_skeleton_training_zh.md:63-90`
- Modify: `docs/semantic_skeleton_ablation.md:51-80`
- Test: repository test suite

**Interfaces:**
- Consumes: the implemented prompt contract.
- Produces: user-facing documentation that matches emitted training and evaluation text.

- [ ] **Step 1: Update both prompt examples and behavior notes**

Replace `Final answer:`, the reference-solution heading/boundaries, bullet-prefixed field guidance, and the old transition with the approved prompt. State that neither the JSON block nor another line exposes `final_answer` in skeleton mode.

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
python -m pytest -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 4: Review final scope**

Run:

```bash
git status --short
git diff -- data_collator.py eval/quick_opsd_common.py tests/test_opsd_skeleton_training.py tests/test_quick_opsd_common.py tests/test_semantic_skeleton_scripts.py docs/opsd_skeleton_training_zh.md docs/semantic_skeleton_ablation.md
```

Expected: only planned files plus the plan document are changed; `.DS_Store` remains untouched as a pre-existing user modification.
