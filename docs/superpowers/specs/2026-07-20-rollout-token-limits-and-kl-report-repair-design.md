# Rollout Token Limits and KL Report Repair Design

**Date:** 2026-07-20

## Goal

Repair the dual-KL experiment so teacher rollout performance and token-length
statistics are not truncated at 1,024 tokens, and make the generated
teacher-base KL HTML report render its interactive curves, heatmap, and spike
table correctly.

## Confirmed Root Causes

`--max-model-len 20000` configures the model's total context window. It does not
configure generation length. The runner currently derives one
`MAX_NEW_TOKENS` value from `--student-tm` and passes it to all four rollout
conditions. With `--student-tm off`, that shared value is 1,024, so all three
teacher conditions are incorrectly truncated at 1,024 tokens.

The HTML report embeds JavaScript in a Python triple-quoted string. Python
interprets `\n` and `\t` before writing the file, producing a newline inside a
JavaScript regular expression and string literal. The browser consequently
stops with `SyntaxError: Invalid regular expression: missing /` before it can
populate the case selector, chart, heatmap, or spike rows.

## Token-Limit Interface

The rollout evaluator will accept independent limits:

- `--student-max-new-tokens`
- `--teacher-max-new-tokens`

The student condition uses the student limit. `teacher_base`,
`teacher_reference`, and `teacher_skeleton` use the teacher limit. The existing
`--max-new-tokens` option remains a backward-compatible all-condition override.
Specific student or teacher options take precedence over the compatibility
option.

The shell runner uses these defaults:

- student with `--student-tm off`: 1,024 new tokens;
- student with `--student-tm on`: 16,384 new tokens;
- all teacher conditions: 16,384 new tokens;
- model context: 20,000 total tokens unless overridden with
  `--max-model-len`.

The runner prints the student limit, teacher limit, and context limit
separately so experiment logs cannot confuse them. Explicit positive-integer
validation will reject invalid values before GPU work begins.

## Report Rendering Repair

The report generator will emit literal JavaScript escape sequences for newline
and tab replacement and for newline joining. Regression tests will inspect the
rendered HTML and fail if Python inserts literal control characters into these
JavaScript expressions again.

The existing experiment directory can be repaired without model inference by
rerunning `eval/quick_teacher_base_kl_report.py` against its existing JSONL and
summary files. This regenerates the HTML, CSV, and spike JSONL artifacts. It
does not change the already truncated rollout data.

## Rerun Requirements

Correct teacher performance and completion-token statistics require a new
rollout run because the missing post-1,024 tokens were never generated. The
normal dual-KL runner will regenerate all four rollouts, both KL probes, the
visualizations, and the optional teacher continuations with the corrected
per-condition limits.

## Documentation

The Chinese experiment runbook is the canonical command reference. It will
state the difference between context length and generation length, document
both new flags and their defaults, provide a full dual-KL command, and provide
an offline report-regeneration command for an existing result directory.

## Verification

1. Unit tests prove each rollout condition receives the intended maximum new
   token count and that the compatibility override still works.
2. Runner tests prove the new flags, defaults, precedence, validation, and
   command forwarding.
3. HTML tests prove the generated JavaScript contains valid escaped newline and
   tab literals.
4. The focused test suite and full test suite pass.
5. The existing `student_teacher_dual_kl_20260720_164159` report is regenerated
   and checked to contain the repaired JavaScript and populated data payload.
