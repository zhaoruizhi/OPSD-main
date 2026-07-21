import sys
import unittest
from unittest.mock import patch


def kl_record(contrast, kl_values, problem_id=7, sample_index=0, case_id="7:0:student"):
    token_texts = [f" token-{index}" for index in range(len(kl_values))]
    top_position = max(range(len(kl_values)), key=lambda index: kl_values[index])
    return {
        "record_type": "kl_contrast",
        "case_id": case_id,
        "problem_id": problem_id,
        "contrast": contrast,
        "target_condition": "student",
        "target_sample_index": sample_index,
        "token_texts": token_texts,
        "kl_per_token": kl_values,
        "delta_logp_target_per_token": [value / 10 for value in kl_values],
        "teacher_entropy_per_token": [1.0 + value / 100 for value in kl_values],
        "student_entropy_per_token": [0.5 for _ in kl_values],
        "top_kl_positions": [
            {
                "position": top_position,
                "teacher_top_tokens": [{"token": " teacher-top", "prob": 0.8}],
                "base_top_tokens": [{"token": " student-top", "prob": 0.7}],
            }
        ],
    }


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = None

    def decode(self, token_ids, skip_special_tokens=False):
        return "|".join(str(token_id) for token_id in token_ids)


class TeacherSpikeSelectionTests(unittest.TestCase):
    def test_select_global_spikes_deduplicates_contrasts_and_ranks_by_max_kl(self):
        from eval.quick_teacher_spike_continuation import select_global_spikes

        records = [
            kl_record("teacher_reference_vs_student", [1.0, 8.0]),
            kl_record("teacher_skeleton_vs_student", [9.0, 2.0]),
        ]

        spikes = select_global_spikes(lambda: iter(records), top_n=2)

        self.assertEqual([(row["position"], row["max_kl"]) for row in spikes], [(0, 9.0), (1, 8.0)])
        self.assertEqual(spikes[0]["reference_kl"], 1.0)
        self.assertEqual(spikes[0]["skeleton_kl"], 9.0)
        self.assertEqual([row["rank"] for row in spikes], [1, 2])

    def test_select_global_spikes_is_global_across_cases(self):
        from eval.quick_teacher_spike_continuation import select_global_spikes

        records = [
            kl_record("teacher_reference_vs_student", [3.0, 1.0], problem_id=7),
            kl_record("teacher_skeleton_vs_student", [2.0, 1.0], problem_id=7),
            kl_record(
                "teacher_reference_vs_student",
                [7.0, 6.0],
                problem_id=8,
                case_id="8:0:student",
            ),
            kl_record(
                "teacher_skeleton_vs_student",
                [4.0, 5.0],
                problem_id=8,
                case_id="8:0:student",
            ),
        ]

        spikes = select_global_spikes(lambda: iter(records), top_n=2)

        self.assertEqual([(row["problem_id"], row["position"]) for row in spikes], [("8", 0), ("8", 1)])

    def test_select_global_spikes_rejects_missing_teacher_contrast(self):
        from eval.quick_teacher_spike_continuation import select_global_spikes

        records = [kl_record("teacher_reference_vs_student", [3.0, 1.0])]

        with self.assertRaisesRegex(ValueError, "reference and skeleton"):
            select_global_spikes(lambda: iter(records), top_n=1)


class TeacherSpikePrefixTests(unittest.TestCase):
    def test_generation_input_stops_before_high_kl_student_token(self):
        from eval.quick_teacher_spike_continuation import build_generation_input_ids

        result = build_generation_input_ids(
            [10, 11],
            [20, 21, 22],
            position=1,
            max_new_tokens=20,
            max_context_tokens=100,
        )

        self.assertEqual(result, [10, 11, 20])

    def test_generation_input_rejects_context_overflow_without_truncation(self):
        from eval.quick_teacher_spike_continuation import build_generation_input_ids

        with self.assertRaisesRegex(ValueError, "exceeds max context"):
            build_generation_input_ids(
                [1] * 90,
                [2] * 20,
                position=19,
                max_new_tokens=20,
                max_context_tokens=100,
            )

    def test_prepare_spike_case_starts_student_display_at_spike(self):
        from eval.quick_teacher_spike_continuation import prepare_spike_case

        spike = {
            "rank": 1,
            "problem_id": "7",
            "sample_index": 0,
            "target_condition": "student",
            "position": 1,
            "max_kl": 9.0,
            "reference_kl": 1.0,
            "skeleton_kl": 9.0,
        }
        rollout = {
            "problem_id": 7,
            "sample_index": 0,
            "condition": "student",
            "problem": "P",
            "solution": "S",
            "completion_token_ids": list(range(30)),
        }

        case = prepare_spike_case(spike, rollout, FakeTokenizer(), display_tokens=20)

        self.assertEqual(case["student_suffix_token_ids"], list(range(1, 21)))
        self.assertEqual(case["student_token_text"], "1")
        self.assertEqual(case["student_prefix_token_count"], 1)

    def test_prepare_spike_case_retokenizes_text_for_legacy_kl_positions(self):
        from eval.quick_teacher_spike_continuation import prepare_spike_case

        class TextTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in text]}

        spike = {
            "rank": 1,
            "problem_id": "7",
            "sample_index": 0,
            "target_condition": "student",
            "target_token_source": "target_tail_text",
            "position": 1,
            "max_kl": 9.0,
            "reference_kl": 1.0,
            "skeleton_kl": 9.0,
        }
        rollout = {
            "problem_id": 7,
            "sample_index": 0,
            "condition": "student",
            "problem": "P",
            "solution": "S",
            "full_generation": "ABC",
            "completion_token_ids": [1, 2, 3],
        }

        case = prepare_spike_case(spike, rollout, TextTokenizer(), display_tokens=2)

        self.assertEqual(case["completion_token_ids"], [65, 66, 67])
        self.assertEqual(case["student_token_text"], "66")

    def test_teacher_input_reuses_condition_prompt_builder_and_exact_prefix(self):
        from eval.quick_teacher_spike_continuation import teacher_input_ids_for_case

        case = {"problem_id": 7, "completion_token_ids": [20, 21, 22], "position": 1}
        with patch(
            "eval.quick_teacher_spike_continuation.context_prompt_ids_for_condition",
            return_value=([10, 11], "reconstructed_prompt_text"),
        ) as prompt_builder:
            input_ids, prompt_count, source = teacher_input_ids_for_case(
                tokenizer=FakeTokenizer(),
                case=case,
                condition="teacher_reference",
                skeletons={},
                max_new_tokens=20,
                max_context_tokens=100,
                teacher_prompt_profile="legacy-20260629",
            )

        self.assertEqual(input_ids, [10, 11, 20])
        self.assertEqual(prompt_count, 2)
        self.assertEqual(source, "reconstructed_prompt_text")
        self.assertEqual(prompt_builder.call_args.kwargs["condition"], "teacher_reference")
        self.assertEqual(
            prompt_builder.call_args.kwargs["teacher_prompt_profile"],
            "legacy-20260629",
        )


class TeacherSpikeReportTests(unittest.TestCase):
    def test_html_report_has_three_columns_and_escapes_model_text(self):
        from eval.quick_teacher_spike_continuation import render_html_report

        record = {
            "rank": 1,
            "problem_id": "7",
            "sample_index": 0,
            "position": 12,
            "max_kl": 9.0,
            "reference_kl": 8.0,
            "skeleton_kl": 9.0,
            "problem": "<problem>",
            "reference_solution": "<solution>",
            "semantic_skeleton": {"key_objects": ["<object>"]},
            "context_before_text": "before ",
            "student_token_text": "<student>",
            "student_suffix_text": "<student> suffix",
            "contrast_metrics": {},
            "continuations": {
                "teacher_reference": {"text": "<reference>"},
                "teacher_skeleton": {"text": "<skeleton>"},
            },
        }

        html = render_html_report([record])

        self.assertIn("Student original", html)
        self.assertIn("Reference teacher", html)
        self.assertIn("Skeleton teacher", html)
        self.assertIn("&lt;student&gt;", html)
        self.assertIn("&lt;reference&gt;", html)
        self.assertNotIn("<student>", html)
        self.assertIn("sample 0", html)

    def test_summary_reports_success_and_failure_counts(self):
        from eval.quick_teacher_spike_continuation import summarize_records

        summary = summarize_records(
            [{"rank": 1, "generation_config": {"max_new_tokens": 20}}]
        )

        self.assertEqual(summary["num_records"], 1)
        self.assertEqual(summary["num_successful_records"], 1)
        self.assertEqual(summary["num_failed_records"], 0)

    def test_cli_defaults_to_global_top_ten_and_twenty_tokens(self):
        from eval.quick_teacher_spike_continuation import parse_args

        argv = [
            "quick_teacher_spike_continuation.py",
            "--base-model",
            "/models/qwen",
            "--kl-file",
            "kl.jsonl",
            "--student-rollout-file",
            "rollout.jsonl",
            "--skeleton-file",
            "skeleton.jsonl",
            "--output-file",
            "output.jsonl",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.top_n, 10)
        self.assertEqual(args.max_new_tokens, 20)


if __name__ == "__main__":
    unittest.main()
