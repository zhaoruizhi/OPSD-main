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

    def test_rollout_parse_args_accepts_legacy_teacher_prompt_profile(self):
        from eval.quick_rollout_openthoughts import parse_args

        argv = [
            "quick_rollout_openthoughts.py",
            "--summary-file",
            "summary.json",
            "--output-file",
            "rollouts.jsonl",
            "--teacher-prompt-profile",
            "legacy-20260629",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.teacher_prompt_profile, "legacy-20260629")

    def test_rollout_user_message_uses_selected_legacy_profile(self):
        from eval.quick_rollout_openthoughts import (
            RolloutConditionSpec,
            user_message_for_rollout,
        )

        prompt = user_message_for_rollout(
            spec=RolloutConditionSpec(
                "teacher_skeleton",
                enable_thinking=True,
                prompt_kind="skeleton",
            ),
            problem="Compute 2+2.",
            solution="2+2=4.",
            skeleton={
                "final_answer": "4",
                "key_objects": [],
                "subgoals": [],
                "critical_intermediates": [],
                "theorem_tags": [],
                "checks": [],
            },
            ground_truth="4",
            problem_id=1,
            teacher_prompt_profile="legacy-20260629",
        )

        self.assertIn("Final answer: 4", prompt)
        self.assertIn('"checks": []', prompt)

    def test_rollout_parse_args_supports_condition_specific_generation_limits(self):
        from eval.quick_rollout_openthoughts import parse_args

        argv = [
            "quick_rollout_openthoughts.py",
            "--summary-file",
            "summary.json",
            "--output-file",
            "rollouts.jsonl",
            "--max-new-tokens",
            "2048",
            "--student-max-new-tokens",
            "4096",
            "--teacher-max-new-tokens",
            "8192",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.max_new_tokens, 2048)
        self.assertEqual(args.student_max_new_tokens, 4096)
        self.assertEqual(args.teacher_max_new_tokens, 8192)

    def test_rollout_generation_limit_prefers_condition_specific_value(self):
        from eval.quick_rollout_openthoughts import max_new_tokens_for_condition

        cases = [
            ("student", 2048, None, None, 2048),
            ("teacher_base", 2048, None, None, 2048),
            ("student", 2048, 4096, 8192, 4096),
            ("teacher_reference", 2048, 4096, 8192, 8192),
            ("teacher_skeleton", 2048, 4096, 8192, 8192),
        ]

        for condition, compatibility, student, teacher, expected in cases:
            with self.subTest(condition=condition, student=student, teacher=teacher):
                self.assertEqual(
                    max_new_tokens_for_condition(
                        condition,
                        max_new_tokens=compatibility,
                        student_max_new_tokens=student,
                        teacher_max_new_tokens=teacher,
                    ),
                    expected,
                )

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
