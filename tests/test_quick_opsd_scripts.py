import sys
import unittest
from unittest.mock import patch


class QuickOpsdScriptShapeTests(unittest.TestCase):
    def test_rollout_condition_specs_match_plan(self):
        from eval.quick_rollout_openthoughts import build_rollout_condition_specs

        specs = {spec.name: spec for spec in build_rollout_condition_specs()}

        self.assertEqual(list(specs), ["student", "teacher_base", "teacher_reference", "teacher_skeleton"])
        self.assertFalse(specs["student"].enable_thinking)
        self.assertEqual(specs["student"].prompt_kind, "student")
        self.assertTrue(specs["teacher_base"].enable_thinking)
        self.assertEqual(specs["teacher_base"].prompt_kind, "base")
        self.assertTrue(specs["teacher_reference"].enable_thinking)
        self.assertEqual(specs["teacher_reference"].prompt_kind, "reference")
        self.assertTrue(specs["teacher_skeleton"].enable_thinking)
        self.assertEqual(specs["teacher_skeleton"].prompt_kind, "skeleton")

    def test_rollout_parse_args_supports_student_thinking_and_checkpoint(self):
        from eval.quick_rollout_openthoughts import parse_args

        argv = [
            "quick_rollout_openthoughts.py",
            "--base-model",
            "/models/Qwen3-1.7B",
            "--checkpoint-dir",
            "/runs/checkpoint-100",
            "--summary-file",
            "summary.json",
            "--output-file",
            "rollouts.jsonl",
            "--student-enable-thinking",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.model, "/models/Qwen3-1.7B")
        self.assertEqual(args.base_model, "/models/Qwen3-1.7B")
        self.assertEqual(args.checkpoint_dir, "/runs/checkpoint-100")
        self.assertTrue(args.student_enable_thinking)

    def test_prefix_condition_specs_match_plan(self):
        from eval.quick_prefix_intervention import build_prefix_condition_specs

        specs = {spec.name: spec for spec in build_prefix_condition_specs()}

        self.assertEqual(
            list(specs),
            [
                "c0_student_continue",
                "c1_prefix_only_teacher",
                "c2_opsd_solution_oracle_teacher",
                "c3_intervention_oracle_teacher",
            ],
        )
        self.assertFalse(specs["c0_student_continue"].enable_thinking)
        self.assertTrue(specs["c3_intervention_oracle_teacher"].enable_thinking)

    def test_logit_context_specs_match_plan(self):
        from eval.quick_logit_probe import build_logit_context_specs

        specs = {spec.name: spec for spec in build_logit_context_specs()}

        self.assertEqual(list(specs), ["student", "teacher_base", "teacher_reference", "teacher_skeleton"])
        self.assertFalse(specs["student"].enable_thinking)
        self.assertEqual(specs["student"].prompt_kind, "student")
        self.assertTrue(specs["teacher_base"].enable_thinking)
        self.assertEqual(specs["teacher_base"].prompt_kind, "base")
        self.assertTrue(specs["teacher_reference"].enable_thinking)
        self.assertEqual(specs["teacher_reference"].prompt_kind, "reference")
        self.assertTrue(specs["teacher_skeleton"].enable_thinking)
        self.assertEqual(specs["teacher_skeleton"].prompt_kind, "skeleton")


if __name__ == "__main__":
    unittest.main()
