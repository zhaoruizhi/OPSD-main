from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest


def make_contrast(contrast, kl_values, top_position):
    teacher_entropies = [0.8, 0.9]
    base_entropies = [0.5, 0.6]
    return {
        "record_type": "kl_contrast",
        "case_id": "7:0:teacher_base",
        "problem_id": 7,
        "target_sample_index": 0,
        "target_condition": "teacher_base",
        "contrast": contrast,
        "token_texts": ["A", " B"],
        "kl_per_token": kl_values,
        "delta_logp_target_per_token": [0.01, -0.02],
        "teacher_entropy_per_token": teacher_entropies,
        "student_entropy_per_token": base_entropies,
        "delta_entropy_per_token": [
            teacher - base for teacher, base in zip(teacher_entropies, base_entropies)
        ],
        "mean_kl": sum(kl_values) / len(kl_values),
        "top1_agreement": 0.5,
        "topk_jaccard": 0.75,
        "mean_delta_logp_target": -0.005,
        "mean_delta_entropy": 0.3,
        "top_kl_positions": [
            {
                "position": top_position,
                "token_text": ["A", " B"][top_position],
                "kl": kl_values[top_position],
                "delta_logp_target": [0.01, -0.02][top_position],
                "teacher_entropy": teacher_entropies[top_position],
                "student_entropy": base_entropies[top_position],
                "delta_entropy": 0.3,
                "teacher_top_tokens": [
                    {"token": "teacher", "prob": 0.7, "logprob": -0.36}
                ],
                "base_top_tokens": [
                    {"token": "base", "prob": 0.6, "logprob": -0.51}
                ],
            }
        ],
    }


class TeacherBaseKlCaseTests(unittest.TestCase):
    def setUp(self):
        self.reference = make_contrast(
            "teacher_reference_vs_teacher_base", [0.1, 0.7], 1
        )
        self.skeleton = make_contrast(
            "teacher_skeleton_vs_teacher_base", [0.2, 0.4], 0
        )
        self.rollouts = [
            {
                "problem_id": 7,
                "sample_index": 0,
                "condition": "teacher_base",
                "problem": "Compute A+B.",
                "ground_truth": "2",
                "predicted_answer": "2",
                "correct": True,
                "formatted": True,
                "finish_reason": "stop",
                "completion_tokens": 2,
                "full_generation": "A B",
            }
        ]
        self.skeletons = {7: {"key_objects": ["A", "B"], "subgoals": ["add"]}}

    def test_pairs_reference_and_skeleton_contrasts_and_builds_spikes(self):
        from eval.quick_teacher_base_kl_report import (
            build_spike_rows,
            build_teacher_base_cases,
        )

        cases = build_teacher_base_cases(
            [self.reference, self.skeleton], self.rollouts, self.skeletons
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["reference_kl"], [0.1, 0.7])
        self.assertEqual(cases[0]["skeleton_kl"], [0.2, 0.4])
        self.assertEqual(cases[0]["problem"], "Compute A+B.")

        spikes = build_spike_rows(cases)

        self.assertEqual([row["position"] for row in spikes], [1, 0])
        self.assertEqual(spikes[0]["reference_kl"], 0.7)
        self.assertEqual(spikes[0]["skeleton_kl"], 0.4)
        self.assertTrue(spikes[0]["saved_for_reference"])
        self.assertFalse(spikes[0]["saved_for_skeleton"])
        self.assertEqual(spikes[0]["reference_teacher_top_tokens"][0]["token"], "teacher")

    def test_rejects_case_missing_one_teacher_contrast(self):
        from eval.quick_teacher_base_kl_report import build_teacher_base_cases

        with self.assertRaisesRegex(ValueError, "requires both"):
            build_teacher_base_cases([self.reference], self.rollouts, self.skeletons)

    def test_rejects_mismatched_per_token_array_lengths(self):
        from eval.quick_teacher_base_kl_report import build_teacher_base_cases

        broken = dict(self.skeleton)
        broken["kl_per_token"] = [0.2]
        with self.assertRaisesRegex(ValueError, "token array lengths"):
            build_teacher_base_cases(
                [self.reference, broken], self.rollouts, self.skeletons
            )


class TeacherBaseKlOutputTests(TeacherBaseKlCaseTests):
    def test_writes_legacy_named_csv_jsonl_and_html_content(self):
        from eval.quick_teacher_base_kl_report import (
            build_teacher_base_cases,
            write_report_outputs,
        )

        cases = build_teacher_base_cases(
            [self.reference, self.skeleton], self.rollouts, self.skeletons
        )
        rollout_summary = {
            "conditions": {
                condition: {
                    "num_problems": 1,
                    "total_generations": 1,
                    "avg_at_n": 1.0,
                    "pass_at_n": 1.0,
                    "majority_vote": 1.0,
                    "format_rate": 1.0,
                    "avg_completion_tokens": token_count,
                }
                for condition, token_count in (
                    ("student", 10.0),
                    ("teacher_base", 20.0),
                    ("teacher_reference", 30.0),
                    ("teacher_skeleton", 40.0),
                )
            }
        }
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            csv_path = output_dir / "teacher_base_kl_reference_vs_skeleton_top_spikes.csv"
            jsonl_path = output_dir / "teacher_base_top_distribution_spikes.jsonl"
            html_path = output_dir / "teacher_base_kl_reference_vs_skeleton_report.html"

            write_report_outputs(
                cases=cases,
                rollout_summary=rollout_summary,
                csv_file=csv_path,
                spikes_jsonl_file=jsonl_path,
                report_file=html_path,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(html_path.exists())
            self.assertIn("reference_kl", csv_path.read_text(encoding="utf-8"))
            spike_records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(any(row["saved_for_skeleton"] for row in spike_records))
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("Teacher Base KL Contrast Visualization", html_text)
            self.assertIn("avg_completion_tokens", html_text)
            self.assertIn(r".replace(/\n/g,'⏎')", html_text)
            self.assertIn(r".replace(/\t/g,'⇥')", html_text)
            self.assertIn(r".join('\n')", html_text)
            self.assertNotIn(".replace(/\n/g", html_text)
            self.assertNotIn(".replace(/\t/g", html_text)
            self.assertNotIn(".join('\n')", html_text)
            for condition in (
                "student",
                "teacher_base",
                "teacher_reference",
                "teacher_skeleton",
            ):
                self.assertIn(condition, html_text)

    def test_parse_args_accepts_all_report_inputs_and_outputs(self):
        from eval.quick_teacher_base_kl_report import parse_args

        argv = [
            "quick_teacher_base_kl_report.py",
            "--logit-file",
            "logits.jsonl",
            "--rollout-file",
            "rollouts.jsonl",
            "--rollout-summary-file",
            "rollout_summary.json",
            "--skeleton-file",
            "skeletons.jsonl",
            "--csv-file",
            "spikes.csv",
            "--spikes-jsonl-file",
            "spikes.jsonl",
            "--report-file",
            "report.html",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.logit_file, "logits.jsonl")
        self.assertEqual(args.rollout_file, "rollouts.jsonl")
        self.assertEqual(args.rollout_summary_file, "rollout_summary.json")
        self.assertEqual(args.skeleton_file, "skeletons.jsonl")
        self.assertEqual(args.csv_file, "spikes.csv")
        self.assertEqual(args.spikes_jsonl_file, "spikes.jsonl")
        self.assertEqual(args.report_file, "report.html")


if __name__ == "__main__":
    unittest.main()
