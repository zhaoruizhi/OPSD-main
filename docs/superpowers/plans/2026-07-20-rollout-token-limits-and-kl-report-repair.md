# Rollout Token Limits and KL Report Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give student and teacher rollouts independent generation limits, repair the broken interactive KL report, and document how to rerun or repair experiments.

**Architecture:** The rollout evaluator will choose a `SamplingParams.max_tokens` value per condition, while the shell runner resolves user-facing defaults and compatibility overrides before launching GPU shards. The HTML repair stays inside the existing report generator and is guarded by rendered-output regression tests. Existing JSONL output can regenerate visualization artifacts without model inference, but corrected performance requires a new rollout.

**Tech Stack:** Python 3, `argparse`, vLLM `SamplingParams`, Bash, `unittest`, HTML/JavaScript.

## Global Constraints

- `--max-model-len` remains the total prompt-plus-completion context limit and defaults to 20,000.
- Student defaults are 1,024 tokens with thinking off and 16,384 tokens with thinking on.
- All teacher rollout conditions default to 16,384 new tokens.
- `--max-new-tokens` remains a backward-compatible override for both student and teacher unless a condition-specific flag overrides it.
- Preserve the user's existing `.DS_Store` modification.
- Do not require model downloads or GPU access in unit tests.

---

### Task 1: Per-condition rollout generation limits

**Files:**
- Modify: `eval/quick_rollout_openthoughts.py:115-220`
- Test: `tests/test_quick_opsd_scripts.py`

**Interfaces:**
- Consumes: parsed `max_new_tokens`, `student_max_new_tokens`, and `teacher_max_new_tokens` values.
- Produces: `max_new_tokens_for_condition(condition_name, max_new_tokens, student_max_new_tokens, teacher_max_new_tokens) -> int` and condition-specific `SamplingParams` instances.

- [ ] **Step 1: Write failing parser and precedence tests**

Add tests that parse both new options and exercise this precedence matrix:

```python
cases = [
    ("student", 2048, None, None, 2048),
    ("teacher_base", 2048, None, None, 2048),
    ("student", 2048, 4096, 8192, 4096),
    ("teacher_skeleton", 2048, 4096, 8192, 8192),
]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_quick_opsd_scripts -v`

Expected: FAIL because the new arguments and resolver do not exist.

- [ ] **Step 3: Implement condition-specific limits**

Add the two optional parser arguments, implement the resolver with specific-over-compatibility precedence, and construct `SamplingParams` inside the condition loop:

```python
max_tokens = max_new_tokens_for_condition(
    spec.name,
    max_new_tokens=args.max_new_tokens,
    student_max_new_tokens=args.student_max_new_tokens,
    teacher_max_new_tokens=args.teacher_max_new_tokens,
)
sampling_params = SamplingParams(..., max_tokens=max_tokens)
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m unittest tests.test_quick_opsd_scripts -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eval/quick_rollout_openthoughts.py tests/test_quick_opsd_scripts.py
git commit -m "fix: separate student and teacher rollout limits"
```

### Task 2: Runner defaults, overrides, and validation

**Files:**
- Modify: `scripts/run_student_teacher_category_kl.sh:14-215`
- Test: `tests/test_quick_opsd_run_script.py`

**Interfaces:**
- Consumes: `--max-new-tokens`, `--student-max-new-tokens`, `--teacher-max-new-tokens`, `--student-tm`, and `--max-model-len`.
- Produces: resolved positive `STUDENT_MAX_NEW_TOKENS` and `TEACHER_MAX_NEW_TOKENS` forwarded to every rollout shard.

- [ ] **Step 1: Write failing runner-shape tests**

Assert that the runner contains both new flags, defaults teacher generation to
16,384, resolves the compatibility override before defaults, prints all three
length limits, validates positive integers, and forwards:

```bash
--student-max-new-tokens "$STUDENT_MAX_NEW_TOKENS"
--teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_quick_opsd_run_script -v`

Expected: FAIL because the runner still forwards one shared token limit.

- [ ] **Step 3: Implement runner resolution**

Parse the new flags. Resolve student and teacher values using this order:

1. condition-specific CLI/environment value;
2. compatibility `MAX_NEW_TOKENS` value;
3. student thinking-mode default or teacher 16,384 default.

Reject empty, zero, negative, or non-integer resolved limits before Phase 0,
then forward both values to the rollout evaluator.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m unittest tests.test_quick_opsd_run_script -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_student_teacher_category_kl.sh tests/test_quick_opsd_run_script.py
git commit -m "fix: configure teacher rollout length independently"
```

### Task 3: Browser-safe KL visualization JavaScript

**Files:**
- Modify: `eval/quick_teacher_base_kl_report.py:425-445`
- Test: `tests/test_teacher_base_kl_report.py`

**Interfaces:**
- Consumes: existing cases, rollout summary, and spike rows.
- Produces: an HTML report whose executable JavaScript contains literal `/\\n/g`, `/\\t/g`, and `.join('\\n')` source sequences.

- [ ] **Step 1: Write the failing rendered-HTML regression test**

After rendering the fixture report, assert:

```python
self.assertIn(r".replace(/\n/g", html_text)
self.assertIn(r".replace(/\t/g", html_text)
self.assertIn(r".join('\n')", html_text)
self.assertNotIn(".replace(/\n/g", html_text)
self.assertNotIn(".join('\n')", html_text)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `python -m unittest tests.test_teacher_base_kl_report.TeacherBaseKlOutputTests.test_writes_legacy_named_csv_jsonl_and_html_content -v`

Expected: FAIL because Python currently converts the escapes to control characters.

- [ ] **Step 3: Escape JavaScript source correctly**

Change the embedded source to preserve the JavaScript escapes:

```javascript
replace(/\\n/g, '⏎').replace(/\\t/g, '⇥')
```

and preserve `\\n` in the distribution join expression.

- [ ] **Step 4: Run report tests and verify pass**

Run: `python -m unittest tests.test_teacher_base_kl_report -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eval/quick_teacher_base_kl_report.py tests/test_teacher_base_kl_report.py
git commit -m "fix: render interactive teacher KL report"
```

### Task 4: Canonical Chinese experiment instructions

**Files:**
- Modify: `docs/experiment_runbook_zh.md:108-215`
- Modify: `docs/student_teacher_category_kl_zh.md:130-245`

**Interfaces:**
- Consumes: the final runner CLI from Tasks 1 and 2.
- Produces: one canonical full command, parameter semantics, expected files, and an offline visualization-repair command.

- [ ] **Step 1: Update the full dual-KL command**

Include:

```bash
--student-max-new-tokens 1024 \
--teacher-max-new-tokens 16384 \
--max-model-len 20000
```

State explicitly that 20,000 is not a completion limit.

- [ ] **Step 2: Document compatibility and TM-on examples**

Explain that `--max-new-tokens N` sets both families for older commands and
that condition-specific flags override it. Show a TM-on example with student
16,384 and an adequately larger total context if the prompt requires it.

- [ ] **Step 3: Document offline report regeneration**

Provide an exact `python eval/quick_teacher_base_kl_report.py` command using
`KL_OUT` and all six required input/output paths. Clarify that this repairs the
visualization only, not truncated rollout performance.

- [ ] **Step 4: Verify command names against code**

Run:

```bash
rg -n "student-max-new-tokens|teacher-max-new-tokens|max-model-len|quick_teacher_base_kl_report" \
  scripts/run_student_teacher_category_kl.sh docs/experiment_runbook_zh.md docs/student_teacher_category_kl_zh.md
```

Expected: both docs match the implemented flags and report filenames.

- [ ] **Step 5: Commit**

```bash
git add docs/experiment_runbook_zh.md docs/student_teacher_category_kl_zh.md
git commit -m "docs: clarify rollout limits and report repair"
```

### Task 5: Regenerate the reported output and complete verification

**Files:**
- Regenerate: `outputs/opsd_quick/student_teacher_dual_kl_20260720_164159/visualizations/teacher_base_kl_reference_vs_skeleton_report.html`
- Regenerate: `outputs/opsd_quick/student_teacher_dual_kl_20260720_164159/visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv`
- Regenerate: `outputs/opsd_quick/student_teacher_dual_kl_20260720_164159/visualizations/teacher_base_top_distribution_spikes.jsonl`

**Interfaces:**
- Consumes: the existing experiment's `logit_probe.jsonl`, `rollouts.jsonl`, `rollout_summary.json`, and `skeletons.jsonl`.
- Produces: repaired local visualization artifacts plus verification evidence; no GPU model execution.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m unittest \
  tests.test_quick_opsd_scripts \
  tests.test_quick_opsd_run_script \
  tests.test_teacher_base_kl_report -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Regenerate the existing visualization**

Run `eval/quick_teacher_base_kl_report.py` with the existing experiment paths
documented in Task 4.

Expected: all three visualization files are replaced successfully without GPU inference.

- [ ] **Step 3: Inspect the repaired payload and JavaScript**

Run a Python read-only check that confirms the report contains non-empty case
and spike payloads and the valid escaped JavaScript sequences.

Expected: ten cases, non-empty spikes, and no broken newline-in-regex sequence.

- [ ] **Step 4: Run the full test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 5: Review final repository state**

Run: `git status --short --branch` and `git diff --check`.

Expected: only the user's pre-existing `.DS_Store` change remains unstaged;
generated ignored output may not appear in Git status.
