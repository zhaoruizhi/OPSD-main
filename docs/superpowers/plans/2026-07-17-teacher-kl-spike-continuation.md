# Teacher KL Spike Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable post-KL experiment that selects the global Top 10 unique KL positions, greedily continues both privileged teachers for 20 tokens, and renders the student/reference/skeleton outputs side by side.

**Architecture:** A streaming JSONL merger first reconstructs valid aggregate inputs. A new Python probe makes a two-pass global spike selection, joins exact student completion token IDs, reconstructs the same teacher prompts used by the KL probe, and generates two deterministic continuations. A shell runner shards the final ten spikes over arbitrary physical GPU IDs and renders merged JSON/summary/HTML outputs.

**Tech Stack:** Python 3, unittest, HuggingFace Transformers, PEFT, PyTorch, Bash, static HTML/CSS.

## Global Constraints

- Branch before student completion token position `p`; teacher input ends at `completion_token_ids[:p]`.
- Rank unique positions by `max(reference_kl, skeleton_kl)` over all cases and contrasts.
- Defaults are `top_n=10`, greedy decoding, and `max_new_tokens=20`.
- Always display student original suffix, reference teacher continuation, and skeleton teacher continuation.
- Reuse `context_prompt_ids_for_condition()` so continuation prompts exactly match KL prompts.
- Accept arbitrary space-separated GPU IDs, including two- and four-GPU configurations.
- Never silently truncate the privileged prompt or student prefix.
- Never overwrite a valid aggregate with a partially merged file.

---

### Task 1: Atomic JSONL shard merger

**Files:**
- Create: `eval/quick_jsonl_merge.py`
- Create: `tests/test_quick_jsonl_merge.py`

**Interfaces:**
- Produces: `iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]`
- Produces: `merge_jsonl_files(input_paths: list[str | Path], output_path: str | Path, sort_key: str | None = None) -> int`
- CLI: repeated `--input-file`, required `--output-file`, optional `--sort-key`

- [ ] **Step 1: Write failing tests for valid merge, optional rank sorting, and corrupt-input atomicity**

```python
def test_merge_jsonl_files_streams_valid_records(self):
    count = merge_jsonl_files([first, second], output)
    self.assertEqual(count, 3)
    self.assertEqual([row["id"] for row in iter_jsonl_records(output)], [1, 2, 3])

def test_merge_jsonl_files_can_sort_small_ranked_outputs(self):
    merge_jsonl_files([first, second], output, sort_key="rank")
    self.assertEqual([row["rank"] for row in iter_jsonl_records(output)], [1, 2, 3])

def test_corrupt_input_does_not_replace_existing_output(self):
    output.write_text('{"kept": true}\n', encoding="utf-8")
    with self.assertRaisesRegex(ValueError, r"bad\.jsonl:2"):
        merge_jsonl_files([bad], output)
    self.assertEqual(output.read_text(encoding="utf-8"), '{"kept": true}\n')
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `python -m unittest tests.test_quick_jsonl_merge -v`

Expected: import failure because `eval.quick_jsonl_merge` does not exist.

- [ ] **Step 3: Implement validated iteration and same-directory atomic replacement**

```python
def iter_jsonl_records(path):
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {source}:{line_number}")
            yield record

def merge_jsonl_files(input_paths, output_path, sort_key=None):
    # Validate every line, write a NamedTemporaryFile in output.parent,
    # fsync it, then os.replace only after all inputs succeed.
    # With sort_key, materialize and sort the small continuation result only.
```

- [ ] **Step 4: Run focused tests and CLI help**

Run: `python -m unittest tests.test_quick_jsonl_merge -v`

Expected: all merger tests pass.

Run: `python eval/quick_jsonl_merge.py --help`

Expected: exit 0 and documented input/output/sort flags.

- [ ] **Step 5: Commit Task 1**

```bash
git add eval/quick_jsonl_merge.py tests/test_quick_jsonl_merge.py
git commit -m "feat: add atomic JSONL shard merger"
```

### Task 2: Global spike selection, exact branching, generation, and HTML report

**Files:**
- Create: `eval/quick_teacher_spike_continuation.py`
- Create: `tests/test_teacher_spike_continuation.py`

**Interfaces:**
- Consumes: `iter_jsonl_records()` from Task 1
- Consumes: `context_prompt_ids_for_condition()` and `lora_adapter_exists()` from `eval.quick_logit_probe`
- Produces: `select_global_spikes(records_factory: Callable[[], Iterable[dict]], top_n: int) -> list[dict[str, Any]]`
- Produces: `prepare_spike_case(spike, rollout, tokenizer, display_tokens) -> dict[str, Any]`
- Produces: `build_generation_input_ids(prompt_ids, completion_ids, position, max_new_tokens, max_context_tokens) -> list[int]`
- Produces: `render_html_report(records: list[dict[str, Any]]) -> str`
- CLI worker mode and `--render-only` mode

- [ ] **Step 1: Write failing selection tests**

```python
def test_select_global_spikes_deduplicates_contrasts_and_ranks_by_max_kl(self):
    records = [reference_record([1.0, 8.0]), skeleton_record([9.0, 2.0])]
    spikes = select_global_spikes(lambda: iter(records), top_n=2)
    self.assertEqual([(s["position"], s["max_kl"]) for s in spikes], [(0, 9.0), (1, 8.0)])
    self.assertEqual(spikes[0]["reference_kl"], 1.0)
    self.assertEqual(spikes[0]["skeleton_kl"], 9.0)

def test_selection_assigns_stable_one_based_ranks(self):
    self.assertEqual([row["rank"] for row in spikes], [1, 2])
```

- [ ] **Step 2: Run selection tests and confirm RED**

Run: `python -m unittest tests.test_teacher_spike_continuation.TeacherSpikeSelectionTests -v`

Expected: import failure because the probe module does not exist.

- [ ] **Step 3: Implement two-pass global selection**

```python
def select_global_spikes(records_factory, top_n):
    # Pass 1: take local top_n from every full kl_per_token array, group by
    # (problem_id, target_sample_index, target_condition, position), rank by max KL.
    # Pass 2: revisit only selected keys and hydrate reference/skeleton per-position
    # KL, delta logp, entropy, and saved top-token distributions.
```

- [ ] **Step 4: Write failing exact-prefix and student-suffix tests**

```python
def test_generation_input_stops_before_high_kl_student_token(self):
    result = build_generation_input_ids([10, 11], [20, 21, 22], position=1,
                                        max_new_tokens=20, max_context_tokens=100)
    self.assertEqual(result, [10, 11, 20])

def test_student_display_begins_at_high_kl_token_and_is_limited_to_twenty(self):
    case = prepare_spike_case(spike_at_1, rollout_with_ids(range(30)), tokenizer, display_tokens=20)
    self.assertEqual(case["student_suffix_token_ids"], list(range(1, 21)))

def test_generation_input_rejects_context_overflow_without_truncation(self):
    with self.assertRaisesRegex(ValueError, "exceeds max context"):
        build_generation_input_ids([1] * 90, [2] * 20, 20, 20, 100)
```

- [ ] **Step 5: Implement case joining and exact branch construction**

```python
def build_generation_input_ids(prompt_ids, completion_ids, position, max_new_tokens, max_context_tokens):
    if position < 0 or position >= len(completion_ids):
        raise ValueError("KL position is outside completion_token_ids")
    input_ids = list(prompt_ids) + list(completion_ids[:position])
    if len(input_ids) + max_new_tokens > max_context_tokens:
        raise ValueError("Teacher continuation input exceeds max context; refusing to truncate")
    return input_ids
```

- [ ] **Step 6: Write failing report and CLI-default tests**

```python
def test_html_report_has_three_columns_and_escapes_model_text(self):
    html = render_html_report([record_with_text("<student>", "<reference>", "<skeleton>")])
    self.assertIn("Student original", html)
    self.assertIn("Reference teacher", html)
    self.assertIn("Skeleton teacher", html)
    self.assertIn("&lt;student&gt;", html)
    self.assertNotIn("<student>", html)

def test_cli_defaults_to_global_top_ten_and_twenty_tokens(self):
    with patch.object(sys, "argv", required_worker_argv):
        args = parse_args()
    self.assertEqual(args.top_n, 10)
    self.assertEqual(args.max_new_tokens, 20)
```

- [ ] **Step 7: Implement HuggingFace/PEFT greedy generation and render-only mode**

```python
generated = model.generate(
    input_ids=input_tensor,
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    use_cache=True,
)
new_token_ids = generated[0, input_tensor.shape[1]:].tolist()
```

Worker records must include both continuation conditions, student suffix, prompt/prefix/generated token counts, finish reason inferred from EOS, full privileged inputs for folded report sections, and global rank. `--render-only` sorts merged records by rank, writes summary JSON, and writes escaped static HTML.

- [ ] **Step 8: Run all Task 2 tests**

Run: `python -m unittest tests.test_teacher_spike_continuation -v`

Expected: all selection, branching, CLI, and HTML tests pass without loading a real model.

- [ ] **Step 9: Commit Task 2**

```bash
git add eval/quick_teacher_spike_continuation.py tests/test_teacher_spike_continuation.py
git commit -m "feat: generate teacher continuations at KL spikes"
```

### Task 3: Arbitrary multi-GPU runner and full-pipeline integration

**Files:**
- Create: `scripts/run_teacher_spike_continuations.sh`
- Modify: `scripts/run_student_teacher_category_kl.sh`
- Modify: `tests/test_quick_opsd_run_script.py`

**Interfaces:**
- Dedicated runner arguments: `--out`, `--base-model`, `--checkpoint-dir`, `--gpu-ids`, `--top-n`, `--max-new-tokens`, `--max-model-len`, `--hf-device-map`
- Full runner additions: `--teacher-continuation-top-n`, `--teacher-continuation-max-new-tokens`, `--skip-teacher-continuations`

- [ ] **Step 1: Add failing shell-shape tests**

```python
def test_teacher_spike_runner_accepts_arbitrary_gpu_ids(self):
    script = Path("scripts/run_teacher_spike_continuations.sh").read_text(encoding="utf-8")
    self.assertIn('--gpu-ids)', script)
    self.assertIn('read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"', script)
    self.assertIn('CUDA_VISIBLE_DEVICES="$gpu"', script)
    self.assertIn('--max-new-tokens "$MAX_NEW_TOKENS"', script)

def test_category_kl_runner_integrates_teacher_spike_phase(self):
    script = Path("scripts/run_student_teacher_category_kl.sh").read_text(encoding="utf-8")
    self.assertIn("run_teacher_spike_continuations.sh", script)
    self.assertIn("--teacher-continuation-max-new-tokens", script)
    self.assertIn("--skip-teacher-continuations", script)
    self.assertIn("quick_jsonl_merge.py", script)
```

- [ ] **Step 2: Run focused shell tests and confirm RED**

Run: `python -m unittest tests.test_quick_opsd_run_script.QuickOpsdRunScriptTests -v`

Expected: failures for missing dedicated runner and missing Phase 3 arguments.

- [ ] **Step 3: Implement the dedicated runner**

The shell flow is exact:

1. Parse model/checkpoint/output/GPU/Top-N/token/context arguments.
2. Build `student_teacher_category_kl_remerged.jsonl` from every available KL shard using `quick_jsonl_merge.py`; when shards are absent, validate the aggregate through a one-input atomic merge.
3. Launch one worker per physical GPU with `CUDA_VISIBLE_DEVICES="$gpu"`, sequential `shard_id`, and total `num_shards`.
4. Merge worker JSONLs with `--sort-key rank` into `teacher_spike_continuations.jsonl`.
5. Call `--render-only` to produce summary and HTML.

- [ ] **Step 4: Integrate Phase 3 and replace raw cat merges in the full runner**

Use `quick_jsonl_merge.py` for rollout and KL aggregates. Invoke the dedicated runner unless `SKIP_TEACHER_CONTINUATIONS=1`, passing the original `GPU_IDS`, model arguments, `TOP_N`, token count, context length, and HF device map.

- [ ] **Step 5: Run tests and shell syntax checks**

Run: `python -m unittest tests.test_quick_opsd_run_script -v`

Expected: all runner-shape tests pass.

Run: `bash -n scripts/run_teacher_spike_continuations.sh scripts/run_student_teacher_category_kl.sh`

Expected: exit 0 with no output.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/run_teacher_spike_continuations.sh scripts/run_student_teacher_category_kl.sh tests/test_quick_opsd_run_script.py
git commit -m "feat: run KL spike continuations on selected GPUs"
```

### Task 4: Chinese experiment guide and complete regression verification

**Files:**
- Create: `docs/teacher_kl_spike_continuations_zh.md`
- Modify: `docs/student_teacher_category_kl_zh.md`

**Interfaces:**
- Documents the add-on run for existing outputs and the full run from scratch.
- Documents two-GPU, four-GPU, arbitrary-ID, base-model-only, and LoRA-checkpoint commands.

- [ ] **Step 1: Write the experiment guide with copy-paste commands**

The guide must include these concrete patterns:

```bash
# Existing result, two GPUs
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/qwen31b_tmon_student_teacher_category_kl_20260709_175323 \
  --base-model /path/to/Qwen3-model \
  --checkpoint-dir /path/to/checkpoint \
  --gpu-ids "0 1" \
  --top-n 10 \
  --max-new-tokens 20

# Four GPUs with explicit physical IDs
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/result \
  --base-model /path/to/Qwen3-model \
  --gpu-ids "4 5 6 7" \
  --top-n 10 \
  --max-new-tokens 20
```

Also explain `--gpu-ids "7 8"`, output paths, exact branching semantics, greedy decoding, automatic shard reconstruction, model/checkpoint consistency, and how to inspect the HTML.

- [ ] **Step 2: Link the new guide from the existing category-KL guide**

Add a section after the output description explaining that the full script now runs Phase 3 by default and that `--skip-teacher-continuations` disables it.

- [ ] **Step 3: Run documentation and code hygiene checks**

Run: `rg -n "gpu-ids|top-n|max-new-tokens|teacher_spike_continuations.html|skip-teacher-continuations" docs/teacher_kl_spike_continuations_zh.md docs/student_teacher_category_kl_zh.md`

Expected: every experiment parameter and output is documented.

Run: `git diff --check`

Expected: exit 0.

- [ ] **Step 4: Run complete relevant regression suite**

Run: `python -m unittest tests.test_quick_jsonl_merge tests.test_teacher_spike_continuation tests.test_quick_logit_probe tests.test_quick_opsd_common tests.test_quick_opsd_run_script tests.test_quick_opsd_scripts -v`

Expected: all tests pass with zero failures/errors.

Run: `python -m compileall -q eval/quick_jsonl_merge.py eval/quick_teacher_spike_continuation.py`

Expected: exit 0.

Run: `bash -n scripts/run_teacher_spike_continuations.sh scripts/run_student_teacher_category_kl.sh`

Expected: exit 0.

- [ ] **Step 5: Commit Task 4**

```bash
git add docs/teacher_kl_spike_continuations_zh.md docs/student_teacher_category_kl_zh.md docs/superpowers/plans/2026-07-17-teacher-kl-spike-continuation.md
git commit -m "docs: explain teacher KL spike experiment"
```
