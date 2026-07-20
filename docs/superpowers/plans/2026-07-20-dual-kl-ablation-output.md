# Dual-KL Ablation Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_student_teacher_category_kl.sh` emit four-condition performance/token-length results, the legacy teacher-base KL comparison, the student-trajectory KL comparison, and teacher continuations in one GPU-selectable run.

**Architecture:** Generate all four rollout conditions once, then run two isolated sharded calls to `quick_logit_probe.py`: one teacher-base-target probe with entropy and one student-target probe without duplicate entropy. Add a stdlib-only report module that pairs the two teacher-base contrasts and emits the three legacy-named visualization artifacts; keep continuations wired only to the student-target probe.

**Tech Stack:** Bash, Python 3 standard library, existing vLLM/Hugging Face evaluators, `unittest`/pytest.

## Global Constraints

- Continue using the current prompt builders; do not restore the old skeleton-as-reference-solution prompt.
- `--gpu-ids` must constrain rollout, both KL probes, and continuation workers.
- Preserve prompt and completion token IDs in rollout records.
- `logit_probe*` means teacher-base trajectory/baseline; `student_teacher_category_kl*` means student trajectory/baseline.
- Teacher continuations consume only `student_teacher_category_kl.jsonl`.
- JSONL merges use `eval/quick_jsonl_merge.py`, not shell concatenation.
- Preserve the user's unrelated `.DS_Store` worktree change.

---

### Task 1: Teacher-base KL report generator

**Files:**
- Create: `eval/quick_teacher_base_kl_report.py`
- Create: `tests/test_teacher_base_kl_report.py`

**Interfaces:**
- Consumes: `logit_probe.jsonl`, `rollouts.jsonl`, `rollout_summary.json`, and `skeletons.jsonl`.
- Produces: `build_teacher_base_cases(...) -> list[dict]`, `build_spike_rows(...) -> list[dict]`, and `write_report_outputs(...) -> None`.
- CLI produces `teacher_base_kl_reference_vs_skeleton_report.html`, `teacher_base_kl_reference_vs_skeleton_top_spikes.csv`, and `teacher_base_top_distribution_spikes.jsonl` at explicitly supplied paths.

- [ ] **Step 1: Write failing pairing and spike tests**

Create fixture records for `teacher_reference_vs_teacher_base` and
`teacher_skeleton_vs_teacher_base` with the same `case_id`. Assert that pairing:

```python
cases = build_teacher_base_cases(records, rollouts, skeletons)
self.assertEqual(len(cases), 1)
self.assertEqual(cases[0]["reference_kl"], [0.1, 0.7])
self.assertEqual(cases[0]["skeleton_kl"], [0.2, 0.4])

spikes = build_spike_rows(cases)
self.assertEqual([row["position"] for row in spikes], [1, 0])
self.assertEqual(spikes[0]["reference_kl"], 0.7)
self.assertEqual(spikes[0]["skeleton_kl"], 0.4)
```

Also assert that a missing contrast or unequal token-array lengths raises
`ValueError` instead of silently producing a partial report.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest tests/test_teacher_base_kl_report.py -q
```

Expected: FAIL because `eval.quick_teacher_base_kl_report` does not exist.

- [ ] **Step 3: Implement record pairing and spike construction**

Implement constants for the two exact contrast names, group contrast records by
`case_id`, join the target `teacher_base` rollout by
`(problem_id, target_sample_index, target_condition)`, and expose per-token
arrays under unambiguous reference/skeleton keys. Build spike rows from the
union of each contrast's `top_kl_positions`, with these legacy fields:

```python
{
    "case_id": case["case_id"],
    "problem_id": case["problem_id"],
    "sample_index": case["sample_index"],
    "target_condition": "teacher_base",
    "position": position,
    "token": case["tokens"][position],
    "reference_kl": case["reference_kl"][position],
    "skeleton_kl": case["skeleton_kl"][position],
    "kl_diff_skeleton_minus_reference": skeleton_kl - reference_kl,
    "abs_kl_diff": abs(skeleton_kl - reference_kl),
    "max_kl": max(reference_kl, skeleton_kl),
    "saved_for_reference": position in reference_top_positions,
    "saved_for_skeleton": position in skeleton_top_positions,
}
```

Hydrate delta-logp, entropy, a nearby token snippet, and the available teacher/base
top-token distributions. Sort descending by `max_kl`, then by stable case/position
keys.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_teacher_base_kl_report.py -q
```

Expected: all pairing/spike tests PASS.

- [ ] **Step 5: Add failing serialization and CLI tests**

Use `TemporaryDirectory` to call `write_report_outputs`. Assert:

```python
self.assertTrue(csv_path.exists())
self.assertTrue(jsonl_path.exists())
self.assertTrue(html_path.exists())
self.assertIn("reference_kl", csv_path.read_text())
self.assertIn('"saved_for_skeleton"', jsonl_path.read_text())
self.assertIn("Teacher Base KL Contrast Visualization", html_path.read_text())
self.assertIn("avg_completion_tokens", html_path.read_text())
```

Test `parse_args()` with explicit logit, rollout, summary, skeleton, CSV, JSONL,
and HTML paths.

- [ ] **Step 6: Run serialization tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_teacher_base_kl_report.py -q
```

Expected: FAIL because output serialization is not implemented.

- [ ] **Step 7: Implement CSV, JSONL, and self-contained HTML output**

Use `csv.DictWriter`, one JSON object per JSONL line, `html.escape`, and an
embedded JSON payload. The HTML must show:

- rollout performance and `avg_completion_tokens` for every available condition;
- global reference/skeleton mean KL;
- per-case reference and skeleton KL token curves or heat strips;
- a sortable top-spike table with top-token distributions.

Do not add plotting or dataframe dependencies.

- [ ] **Step 8: Verify report tests and commit**

Run:

```bash
python3 -m pytest tests/test_teacher_base_kl_report.py -q
```

Expected: PASS.

Commit:

```bash
git add eval/quick_teacher_base_kl_report.py tests/test_teacher_base_kl_report.py
git commit -m "feat: generate teacher-base KL comparison reports"
```

---

### Task 2: Four-condition rollout and two isolated KL probes

**Files:**
- Modify: `tests/test_quick_opsd_run_script.py`
- Modify: `scripts/run_student_teacher_category_kl.sh`

**Interfaces:**
- Consumes: the existing runner flags, including `--gpu-ids`, `--student-tm`, `--sample-size`, `--trajectory-sample-index`, and continuation controls.
- Produces: four-condition `rollouts.jsonl`/`rollout_summary.json`, teacher-base `logit_probe*`, and student-target `student_teacher_category_kl*`.
- Reuses: `eval/quick_jsonl_merge.py` for every shard merge.

- [ ] **Step 1: Replace the old single-probe script test with failing dual-probe assertions**

Assert the script contains all of the following:

```python
self.assertIn('--skeleton-file "$OUT/skeletons.jsonl"', script)
self.assertNotIn("--condition student", script)
self.assertIn('rollout_shard${gpu}.jsonl', script)
self.assertIn('rollouts.jsonl', script)
self.assertIn('rollout_summary.json', script)
self.assertIn("--trajectory-condition teacher_base", script)
self.assertIn("--baseline-condition teacher_base", script)
self.assertIn('logit_probe_shard${gpu}.jsonl', script)
self.assertIn('logit_summary.json', script)
self.assertIn("--trajectory-condition student", script)
self.assertIn("--baseline-condition student", script)
self.assertIn('student_teacher_category_kl_shard${gpu}.jsonl', script)
```

Check ordering so teacher-base KL completes before student KL and continuation,
and check only the student probe contains `--skip-rollout-entropy`.

- [ ] **Step 2: Run the focused runner test and verify RED**

Run:

```bash
python3 -m pytest tests/test_quick_opsd_run_script.py -q
```

Expected: FAIL because the runner is still student-only and has no teacher-base
probe.

- [ ] **Step 3: Change Phase 1 to run all current-prompt conditions**

Remove `--condition student`, add
`--skeleton-file "$OUT/skeletons.jsonl"`, and rename shard outputs to
`rollout_shard${gpu}.jsonl`. Merge atomically to `rollouts.jsonl`, then summarize
to `rollout_summary.json`. Keep `"${STUDENT_THINKING_ARGS[@]}"`; the rollout
module applies it only to the student condition while all teacher condition
specs keep their current thinking/prompt settings.

- [ ] **Step 4: Add the teacher-base KL phase**

Launch one worker per selected GPU with:

```bash
--rollout-file "$OUT/rollouts.jsonl"
--skeleton-file "$OUT/skeletons.jsonl"
--trajectory-condition teacher_base
--baseline-condition teacher_base
--teacher-condition teacher_reference
--teacher-condition teacher_skeleton
--require-context-rollouts
```

Do not pass `--skip-rollout-entropy`. Merge to `logit_probe.jsonl`, summarize to
`logit_summary.json`, and retain shard files.

- [ ] **Step 5: Keep the student KL phase isolated**

Point the existing student probe at `rollouts.jsonl`, add
`--require-context-rollouts`, retain `--skip-rollout-entropy`, and preserve all
`student_teacher_category_kl*` filenames. Use separate shell arrays for the two
sets of shard files so the merge inputs cannot cross.

- [ ] **Step 6: Verify shell and focused tests are GREEN**

Run:

```bash
bash -n scripts/run_student_teacher_category_kl.sh
python3 -m pytest tests/test_quick_opsd_run_script.py -q
```

Expected: syntax check and all focused tests PASS.

- [ ] **Step 7: Commit the runner change**

```bash
git add scripts/run_student_teacher_category_kl.sh tests/test_quick_opsd_run_script.py
git commit -m "feat: run four rollouts and dual KL probes"
```

---

### Task 3: Integrate legacy reports and preserve student-spike continuations

**Files:**
- Modify: `tests/test_quick_opsd_run_script.py`
- Modify: `scripts/run_student_teacher_category_kl.sh`

**Interfaces:**
- Consumes: Task 1 report CLI and Task 2 outputs.
- Produces: three old-compatible visualization paths plus the existing continuation paths.

- [ ] **Step 1: Add failing integration assertions**

Assert the runner invokes `quick_teacher_base_kl_report.py` after
`logit_summary.json` is built with exactly these outputs:

```text
visualizations/teacher_base_kl_reference_vs_skeleton_report.html
visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv
visualizations/teacher_base_top_distribution_spikes.jsonl
```

Also assert the continuation call still uses:

```bash
--kl-file "$OUT/student_teacher_category_kl.jsonl"
--student-rollout-file "$OUT/rollouts.jsonl"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest tests/test_quick_opsd_run_script.py -q
```

Expected: FAIL because the report phase and explicit all-rollout input are absent.

- [ ] **Step 3: Wire the report and continuation phases**

Call the report generator with the merged teacher-base logit file, all-condition
rollout file, rollout summary, skeleton file, and three explicit legacy output
paths. Then call `run_teacher_spike_continuations.sh` with the student KL file
and explicit `--student-rollout-file "$OUT/rollouts.jsonl"`.

Update terminal completion output to list:

- rollout summary;
- teacher-base KL summary/report;
- student-target KL summary;
- continuation report when enabled.

- [ ] **Step 4: Verify and commit**

Run:

```bash
bash -n scripts/run_student_teacher_category_kl.sh
python3 -m pytest tests/test_quick_opsd_run_script.py tests/test_teacher_base_kl_report.py -q
```

Expected: PASS.

Commit:

```bash
git add scripts/run_student_teacher_category_kl.sh tests/test_quick_opsd_run_script.py
git commit -m "feat: retain KL reports and teacher continuations"
```

---

### Task 4: Document exact server commands and output interpretation

**Files:**
- Modify: `docs/experiment_runbook_zh.md`
- Modify: `docs/student_teacher_category_kl_zh.md`

**Interfaces:**
- Documents: one-shot dual-KL plus continuation, dual-KL without continuation,
  and continuation-only resume commands.

- [ ] **Step 1: Update the main runbook**

Use the server paths supplied by the user and show a command with:

```bash
KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --skeleton-file /home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl \
  --out "$KL_OUT" \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 200 \
  --seed 0
```

Explain that `rollout_summary.json` contains performance and
`avg_completion_tokens` for four conditions; `logit_summary.json` is the
teacher-base comparison; `student_teacher_category_kl_summary.json` is the
student-trajectory comparison used for continuation.

- [ ] **Step 2: Document skip and resume commands**

For dual KL without continuation, append `--skip-teacher-continuations`. For a
later continuation-only run, document:

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --out "$KL_OUT" \
  --student-rollout-file "$KL_OUT/rollouts.jsonl" \
  --skeleton-file "$KL_OUT/skeletons.jsonl" \
  --gpu-ids "4 5" \
  --top-n 10 \
  --max-new-tokens 200 \
  --max-model-len 20000 \
  --hf-device-map cuda
```

- [ ] **Step 3: Correct the focused KL guide**

Replace student-only wording with a precise phase table and list all old and
new output paths. State explicitly that both teacher probes use the current
reference/skeleton prompt builders and that the skeleton prompt is style-neutral.

- [ ] **Step 4: Validate docs and commit**

Run:

```bash
rg -n "rollout_summary.json|logit_summary.json|student_teacher_category_kl_summary.json|teacher_spike_continuations.html|gpu-ids" docs/experiment_runbook_zh.md docs/student_teacher_category_kl_zh.md
```

Expected: each artifact family and GPU option appears in the intended command
and interpretation sections.

Commit:

```bash
git add docs/experiment_runbook_zh.md docs/student_teacher_category_kl_zh.md
git commit -m "docs: explain dual KL experiment workflow"
```

---

### Task 5: Regression verification

**Files:**
- Verify only; modify a test or implementation file only if a demonstrated regression requires it.

**Interfaces:**
- Confirms: syntax, prompt invariants, report schema, dual-probe orchestration, and existing continuation behavior.

- [ ] **Step 1: Run the targeted suite**

```bash
python3 -m pytest \
  tests/test_quick_opsd_run_script.py \
  tests/test_teacher_base_kl_report.py \
  tests/test_quick_logit_probe.py \
  tests/test_teacher_spike_continuation.py \
  tests/test_quick_opsd_scripts.py \
  tests/test_semantic_skeleton_scripts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run static validation**

```bash
bash -n scripts/run_student_teacher_category_kl.sh scripts/run_teacher_spike_continuations.sh
python3 -m py_compile eval/quick_teacher_base_kl_report.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run an offline fixture report smoke test**

Use the unit-test fixture data to render all three report outputs in a temporary
directory; verify that each file is non-empty and that the HTML contains all
four rollout condition names. This test must not download a model or require a
GPU.

- [ ] **Step 4: Inspect final scope**

```bash
git status --short
git log --oneline -6
```

Expected: only the pre-existing `.DS_Store` remains uncommitted, and the plan,
implementation, tests, and docs appear in focused commits.

