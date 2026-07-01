import unittest


class SemanticSkeletonScriptTests(unittest.TestCase):
    def test_extract_indices_from_rollouts_filters_condition_and_sorts_unique_ids(self):
        from eval.prepare_sample_manifest import extract_indices_from_rollouts

        records = [
            {"condition": "teacher_reference", "problem_id": 9},
            {"condition": "student_full", "problem_id": 5, "sample_index": 1},
            {"condition": "student_full", "problem_id": 3, "sample_index": 0},
            {"condition": "student_full", "problem_id": 5, "sample_index": 0},
        ]

        self.assertEqual(extract_indices_from_rollouts(records, "student_full"), [3, 5])

    def test_build_manifest_records_dataset_split_seed_and_indices(self):
        from eval.prepare_sample_manifest import build_manifest

        manifest = build_manifest(
            dataset="dataset/name",
            split="train",
            sample_size=2,
            seed=11,
            indices=[7, 2],
        )

        self.assertEqual(
            manifest,
            {
                "dataset": "dataset/name",
                "split": "train",
                "sample_size": 2,
                "seed": 11,
                "indices": [2, 7],
            },
        )

    def test_skeleton_prompt_uses_answer_and_reference_without_problem(self):
        from eval.generate_semantic_skeletons import build_skeleton_compiler_prompt

        prompt = build_skeleton_compiler_prompt(
            answer="4",
            reference_solution="Compute 2+2 and conclude.",
        )

        self.assertIn("ANSWER:\n4", prompt)
        self.assertIn("REFERENCE_SOLUTION:\nCompute 2+2 and conclude.", prompt)
        self.assertNotIn("PROBLEM:", prompt)

    def test_parse_skeleton_response_requires_valid_json_and_normalizes_aliases(self):
        from eval.generate_semantic_skeletons import parse_skeleton_response

        skeleton = parse_skeleton_response(
            '{"final_answer":"4","key_objects":[],"subgoals":[],"critical_intermediate":["2+2=4"],'
            '"theorem_tags":[],"check":["box the answer"]}'
        )

        self.assertEqual(skeleton["critical_intermediates"], ["2+2=4"])
        self.assertEqual(skeleton["checks"], ["box the answer"])

    def test_reference_prompt_keeps_solution_and_exposes_final_answer(self):
        from eval.quick_opsd_common import build_reference_user_message

        prompt = build_reference_user_message(
            problem="Compute 2+2.",
            solution="A reference solution says 2+2=4.",
            answer="4",
        )

        self.assertIn("A reference solution says 2+2=4.", prompt)
        self.assertIn("Final answer: 4", prompt)
        self.assertIn("Please reason step by step, and put your final answer within \\boxed{}.", prompt)

    def test_skeleton_prompt_removes_final_answer_from_skeleton_and_exposes_answer_once(self):
        from eval.quick_opsd_common import build_semantic_skeleton_user_message

        prompt = build_semantic_skeleton_user_message(
            problem="Compute 2+2.",
            skeleton={
                "final_answer": "4",
                "key_objects": [],
                "subgoals": ["establish the sum"],
                "critical_intermediates": ["2+2=4"],
                "theorem_tags": [],
                "checks": [],
            },
            answer="4",
        )

        self.assertIn("Here is a style-neutral semantic skeleton extracted from a reference solution:", prompt)
        self.assertIn("=== Semantic Skeleton Begin ===", prompt)
        self.assertIn("=== Semantic Skeleton End ===", prompt)
        self.assertIn("semantic skeleton", prompt.lower())
        skeleton_block = prompt.split("=== Semantic Skeleton Begin ===\n", 1)[1].split(
            "\n=== Semantic Skeleton End ===", 1
        )[0]
        self.assertNotIn("final_answer", skeleton_block)
        self.assertIn("Final answer: 4", prompt)
        self.assertIn("establish the sum", skeleton_block)


if __name__ == "__main__":
    unittest.main()
