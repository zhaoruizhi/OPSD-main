# Semantic Skeleton Prompt Design

## Goal

Make the skeleton-conditioned teacher prompt explicitly describe its privileged context as a style-neutral semantic skeleton instead of presenting the JSON as a reference solution. Keep the reference-conditioned teacher prompt unchanged.

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

## Testing

Use a red-green cycle. First update the skeleton prompt tests and run them against the old implementation to verify failures caused by the missing semantic-skeleton wording. Then update the implementation and run the focused test files. Finally run the full test suite to detect unrelated prompt-shape dependencies.
