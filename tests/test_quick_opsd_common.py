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
        self.assertIn("Hidden diagnostic information", intervention)
        self.assertIn(json.dumps("reach 4")[1:-1], intervention)
        self.assertIn("Do not restart", intervention)

    def test_semantic_skeleton_prompt_uses_structured_reference_only(self):
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

        self.assertIn("Problem: 2+2?", prompt)
        self.assertIn("Here is a style-neutral semantic skeleton extracted from a reference solution:", prompt)
        self.assertIn("Semantic Skeleton Begin", prompt)
        self.assertIn("Semantic Skeleton End", prompt)
        self.assertIn('"critical_intermediates"', prompt)
        self.assertIn('"checks"', prompt)
        self.assertIn("semantic skeleton", prompt.lower())

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
