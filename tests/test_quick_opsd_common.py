import json
import unittest

from eval.quick_opsd_common import (
    build_reference_user_message,
    build_intervention_user_message,
    build_opsd_oracle_user_message,
    build_semantic_skeleton_user_message,
    build_student_user_message,
    choose_stratified_indices,
    continuation_metrics,
    detect_restart,
    extract_boxed_answer,
    normalize_semantic_skeleton,
    ngram_overlap_rate,
    summarize_token_category_values,
    token_category,
    read_skeleton_file,
    shard_items,
    split_prefix_by_token_ratio,
    write_jsonl,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class QuickOpsdCommonTests(unittest.TestCase):
    def test_extract_boxed_answer_handles_nested_braces(self):
        self.assertEqual(extract_boxed_answer(r"work \boxed{\frac{1}{2}} done"), r"\frac{1}{2}")

    def test_choose_stratified_indices_is_deterministic_and_balanced(self):
        rows = [{"generated_token_count": i} for i in range(90)]

        first = choose_stratified_indices(rows, sample_size=12, seed=7)
        second = choose_stratified_indices(rows, sample_size=12, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(any(i < 30 for i in first))
        self.assertTrue(any(30 <= i < 60 for i in first))
        self.assertTrue(any(i >= 60 for i in first))

    def test_shard_items_uses_modulo_partition_without_overlap(self):
        shards = [shard_items(list(range(10)), shard_id=i, num_shards=4) for i in range(4)]
        flattened = sorted(item for shard in shards for item in shard)

        self.assertEqual(flattened, list(range(10)))
        self.assertEqual(shards[0], [0, 4, 8])

    def test_prompt_builders_include_expected_context_boundaries(self):
        student = build_student_user_message("2+2?")
        oracle = build_reference_user_message("2+2?", "Because 4.")
        intervention = build_intervention_user_message(
            "2+2?",
            {
                "validity": "uncertain",
                "first_invalid_span": "unknown",
                "local_reason": "quick check",
                "minimal_repair_hint": "reach 4",
                "next_local_subgoal": "finish arithmetic",
            },
        )

        self.assertIn("Problem: 2+2?", student)
        self.assertIn("Reference Solution Begin", oracle)
        self.assertIn("Because 4.", oracle)
        self.assertEqual(oracle, build_opsd_oracle_user_message("2+2?", "Because 4."))
        self.assertNotIn("Interpret the fields as follows:", oracle)
        self.assertIn("Hidden diagnostic information", intervention)
        self.assertIn(json.dumps("reach 4")[1:-1], intervention)
        self.assertIn("Do not restart", intervention)

    def test_semantic_skeleton_prompt_explains_fields_before_reasoning_instruction(self):
        prompt = build_semantic_skeleton_user_message(
            "2+2?",
            {
                "final_answer": "4",
                "key_objects": [{"name": "x", "constraints": ["integer"]}],
                "subgoals": ["Evaluate the sum"],
                "critical_intermediates": ["2+2=4"],
                "theorem_tags": ["arithmetic"],
                "checks": ["avoid copying prose"],
            },
        )

        self.assertIn("Problem: 2+2?\n", prompt)
        self.assertIn(
            "Below is a style-neutral semantic skeleton extracted from a reference solution.\n"
            "=== Semantic Skeleton Begin ===",
            prompt,
        )
        self.assertIn("Semantic Skeleton Begin", prompt)
        self.assertIn("Semantic Skeleton End", prompt)
        self.assertNotIn("Final answer:", prompt)
        self.assertNotIn("Reference Solution Begin", prompt)
        self.assertNotIn("Here is a reference solution to this problem:", prompt)
        skeleton_block = prompt.split("=== Semantic Skeleton Begin ===\n", 1)[1].split(
            "\n=== Semantic Skeleton End ===", 1
        )[0]
        self.assertIn('"critical_intermediates"', skeleton_block)
        self.assertIn('"check"', skeleton_block)
        self.assertNotIn('"checks"', skeleton_block)

        field_guidance = (
            'Interpret the fields as follows:\n'
            '"key_objects" records potentially important mathematical objects and constraints.\n'
            '"subgoals" records possible mathematical objectives.\n'
            '"critical_intermediates" records potentially useful mathematical checkpoints. '
            'They are not mandatory generated sentences and do not imply that the reference path is the only valid path.\n'
            '"theorem_tags" records optional and non-exclusive methods. Do not force a listed theorem when another valid approach is more natural.\n'
            '"check" records validity conditions or possible failure modes. Apply a check only when it is relevant to the reasoning being used.'
        )
        self.assertIn(f"=== Semantic Skeleton End ===\n{field_guidance}\nAfter reading", prompt)
        self.assertIn(
            "After reading the reference solution above, make sure you truly understand the reasoning. "
            "Now, using your own words",
            prompt,
        )
        self.assertNotIn("reasoning behind each step — do not copy or paraphrase it", prompt)

    def test_normalize_semantic_skeleton_accepts_legacy_aliases(self):
        skeleton = normalize_semantic_skeleton(
            {
                "final_answer": "6",
                "key_objects": [{"name": "x", "constraints": ["real"]}],
                "subgoals": ["Use Vieta"],
                "critical_intermediate": ["x_1+x_2=5"],
                "theorem_tags": ["Vieta"],
                "check": ["verify signs"],
            }
        )

        self.assertEqual(skeleton["critical_intermediates"], ["x_1+x_2=5"])
        self.assertEqual(skeleton["checks"], ["verify signs"])
        self.assertNotIn("critical_intermediate", skeleton)
        self.assertNotIn("check", skeleton)

    def test_read_skeleton_file_keeps_successful_normalized_entries(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "skeletons.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "problem_id": 7,
                        "status": "ok",
                        "skeleton": {
                            "final_answer": "4",
                            "key_objects": [],
                            "subgoals": [],
                            "critical_intermediate": ["2+2=4"],
                            "theorem_tags": [],
                            "check": ["box final answer"],
                        },
                    },
                    {"problem_id": 8, "status": "error", "skeleton": None},
                ],
            )

            skeletons = read_skeleton_file(path)

        self.assertEqual(set(skeletons), {7})
        self.assertEqual(skeletons[7]["critical_intermediates"], ["2+2=4"])
        self.assertEqual(skeletons[7]["checks"], ["box final answer"])

    def test_split_prefix_by_token_ratio_decodes_middle_prefix(self):
        prefix, tail, cutoff = split_prefix_by_token_ratio(FakeTokenizer(), "a b c d e", ratio=0.5)

        self.assertEqual(prefix, "a b")
        self.assertEqual(tail, "c d e")
        self.assertEqual(cutoff, 2)

    def test_restart_and_copy_metrics(self):
        self.assertTrue(detect_restart("Let's start over and solve from scratch."))
        self.assertFalse(detect_restart("Thus the next value is 4."))
        self.assertGreater(ngram_overlap_rate("a b c d x", "a b c d y", n=4), 0.0)

    def test_token_category_uses_style_math_and_other_lists(self):
        self.assertEqual(token_category(" wait"), "style")
        self.assertEqual(token_category("Therefore,"), "style")
        self.assertEqual(token_category(" fraction"), "math")
        self.assertEqual(token_category("variable"), "math")
        self.assertEqual(token_category("in"), "math")
        self.assertEqual(token_category("x^2"), "other")
        self.assertEqual(token_category("2"), "other")
        self.assertEqual(token_category("="), "other")
        self.assertEqual(token_category(r"\frac"), "other")
        self.assertEqual(token_category(" elephant"), "other")

    def test_summarize_token_category_values_uses_token_weighted_means(self):
        summary = summarize_token_category_values(
            [" wait", " fraction", "variable", "x^2", "plain"],
            [0.8, 0.2, 0.4, 0.5, 0.1],
        )

        self.assertEqual(summary["style"]["num_tokens"], 1)
        self.assertAlmostEqual(summary["style"]["mean_kl"], 0.8)
        self.assertEqual(summary["math"]["num_tokens"], 2)
        self.assertAlmostEqual(summary["math"]["mean_kl"], 0.3)
        self.assertEqual(summary["other"]["num_tokens"], 2)
        self.assertAlmostEqual(summary["other"]["mean_kl"], 0.3)

    def test_continuation_metrics_marks_correctness_and_copy_rate(self):
        metrics = continuation_metrics(
            prefix="First compute x=2.",
            continuation=" Continue locally and finish with \\boxed{4}.",
            ground_truth="4",
            reference_solution="First compute x=2. Continue locally and finish with 4.",
        )

        self.assertTrue(metrics["formatted"])
        self.assertTrue(metrics["correct"])
        self.assertFalse(metrics["restart"])
        self.assertTrue(metrics["prefix_preserved"])
        self.assertGreater(metrics["notation_consistency"], 0.0)
        self.assertGreater(metrics["locality_score"], 0.0)
        self.assertGreater(metrics["reference_copy_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
