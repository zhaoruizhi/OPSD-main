import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.quick_opsd_common import summarize_logit_records
from eval.quick_logit_probe import (
    build_rollout_entropy_record,
    build_top_kl_positions,
    compare_contexts,
    completed_logit_record_keys,
    context_prompt_ids_for_condition,
    contrast_specs_from_args,
    select_full_response_cases,
    shard_cases,
    target_token_ids_for_case,
    truncate_target_text,
)


class QuickLogitProbeTests(unittest.TestCase):
    def test_parse_args_defaults_to_full_response_rollout_probe(self):
        from eval.quick_logit_probe import parse_args

        argv = [
            "quick_logit_probe.py",
            "--rollout-file",
            "rollouts.jsonl",
            "--output-file",
            "logit_probe.jsonl",
            "--summary-file",
            "logit_summary.json",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.rollout_file, "rollouts.jsonl")
        self.assertIsNone(args.prefix_file)
        self.assertEqual(args.probe_tokens, 0)
        self.assertEqual(args.trajectory_condition, ["teacher_base"])
        self.assertEqual(args.trajectory_sample_index, 0)
        self.assertFalse(args.skip_rollout_entropy)
        self.assertEqual(args.hf_device_map, "cuda")
        self.assertEqual(args.shard_id, 0)
        self.assertEqual(args.num_shards, 1)
        self.assertFalse(hasattr(args, "score_batch_size"))
        self.assertFalse(hasattr(args, "gpu_memory_utilization"))
        self.assertFalse(hasattr(args, "tensor_parallel_size"))

    def test_parse_args_supports_contrast_only_fast_path(self):
        from eval.quick_logit_probe import parse_args

        argv = [
            "quick_logit_probe.py",
            "--rollout-file",
            "rollouts.jsonl",
            "--output-file",
            "logit_probe.jsonl",
            "--summary-file",
            "logit_summary.json",
            "--skip-rollout-entropy",
            "--hf-device-map",
            "auto",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.skip_rollout_entropy)
        self.assertEqual(args.hf_device_map, "auto")

    def test_parse_args_supports_student_teacher_category_kl(self):
        from eval.quick_logit_probe import parse_args

        argv = [
            "quick_logit_probe.py",
            "--base-model",
            "/models/Qwen3-1.7B",
            "--checkpoint-dir",
            "/runs/checkpoint-100",
            "--rollout-file",
            "rollouts.jsonl",
            "--output-file",
            "student_teacher_kl.jsonl",
            "--summary-file",
            "student_teacher_kl_summary.json",
            "--trajectory-condition",
            "student",
            "--baseline-condition",
            "student",
            "--teacher-condition",
            "teacher_reference",
            "--teacher-condition",
            "teacher_skeleton",
            "--student-enable-thinking",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.model, "/models/Qwen3-1.7B")
        self.assertEqual(args.base_model, "/models/Qwen3-1.7B")
        self.assertEqual(args.checkpoint_dir, "/runs/checkpoint-100")
        self.assertEqual(args.trajectory_condition, ["student"])
        self.assertEqual(args.baseline_condition, "student")
        self.assertEqual(args.teacher_condition, ["teacher_reference", "teacher_skeleton"])
        self.assertTrue(args.student_enable_thinking)
        self.assertEqual(
            contrast_specs_from_args(args),
            [
                ("teacher_reference_vs_student", "student", "teacher_reference"),
                ("teacher_skeleton_vs_student", "student", "teacher_skeleton"),
            ],
        )

    def test_select_full_response_cases_filters_to_requested_sample_index(self):
        records = [
            {
                "problem_id": 1,
                "condition": "teacher_base",
                "sample_index": 0,
                "full_generation": "base one sample zero",
            },
            {
                "problem_id": 1,
                "condition": "teacher_base",
                "sample_index": 1,
                "full_generation": "base one sample one",
            },
            {
                "problem_id": 2,
                "condition": "teacher_base",
                "sample_index": 0,
                "full_generation": "base two sample zero",
            },
        ]

        cases = select_full_response_cases(
            records,
            logit_size=0,
            seed=0,
            trajectory_conditions=["teacher_base"],
            trajectory_sample_index=0,
        )

        self.assertEqual([case["case_id"] for case in cases], ["1:0:teacher_base", "2:0:teacher_base"])
        self.assertEqual([case["target_tail_text"] for case in cases], ["base one sample zero", "base two sample zero"])

    def test_select_full_response_cases_attaches_context_records(self):
        records = [
            {
                "problem_id": 1,
                "condition": "teacher_base",
                "sample_index": 0,
                "full_generation": "base",
            },
            {
                "problem_id": 1,
                "condition": "teacher_reference",
                "sample_index": 0,
                "full_generation": "reference",
            },
            {
                "problem_id": 1,
                "condition": "teacher_skeleton",
                "sample_index": 0,
                "full_generation": "skeleton",
            },
        ]

        cases = select_full_response_cases(
            records,
            logit_size=0,
            seed=0,
            trajectory_conditions=["teacher_base"],
            trajectory_sample_index=0,
            context_conditions=["teacher_base", "teacher_reference", "teacher_skeleton"],
        )

        self.assertEqual(len(cases), 1)
        context_records = cases[0]["context_records"]
        self.assertEqual(context_records["teacher_base"]["full_generation"], "base")
        self.assertEqual(context_records["teacher_reference"]["full_generation"], "reference")
        self.assertEqual(context_records["teacher_skeleton"]["full_generation"], "skeleton")

    def test_shard_cases_partitions_cases_without_overlap(self):
        cases = [{"case_id": f"case-{index}"} for index in range(10)]

        shards = [shard_cases(cases, shard_id=index, num_shards=4) for index in range(4)]

        self.assertEqual(shards[0], [cases[0], cases[4], cases[8]])
        self.assertEqual(shards[1], [cases[1], cases[5], cases[9]])
        self.assertEqual(shards[2], [cases[2], cases[6]])
        self.assertEqual(shards[3], [cases[3], cases[7]])
        self.assertEqual(
            sorted(item["case_id"] for shard in shards for item in shard),
            [f"case-{index}" for index in range(10)],
        )

    def test_completed_logit_record_keys_supports_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "logit_probe.jsonl"
            output_path.write_text(
                "\n".join(
                    [
                        '{"record_type":"kl_contrast","case_id":"1:0:student","contrast":"teacher_reference_vs_student"}',
                        '{"record_type":"rollout_entropy","case_id":"1:0:teacher_reference","condition":"teacher_reference"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            keys = completed_logit_record_keys(output_path)

        self.assertEqual(
            keys,
            {
                "kl_contrast:1:0:student:teacher_reference_vs_student",
                "rollout_entropy:1:0:teacher_reference:teacher_reference",
            },
        )

    def test_truncate_target_text_retokenizes_text(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(char) for char in text]}

            def decode(self, token_ids, skip_special_tokens=False):
                return "".join(chr(token_id) for token_id in token_ids)

        text, token_ids = truncate_target_text(FakeTokenizer(), "abcd", probe_tokens=2)

        self.assertEqual(text, "ab")
        self.assertEqual(token_ids, [97, 98])

    def test_target_token_ids_prefers_stored_completion_ids(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                raise AssertionError("stored completion ids should avoid text tokenization")

            def decode(self, token_ids, skip_special_tokens=False):
                return "".join(chr(token_id) for token_id in token_ids)

        text, token_ids, source = target_token_ids_for_case(
            FakeTokenizer(),
            {
                "target_tail_text": "abcd",
                "completion_token_ids": [120, 121, 122],
            },
            probe_tokens=2,
        )

        self.assertEqual(text, "xy")
        self.assertEqual(token_ids, [120, 121])
        self.assertEqual(source, "completion_token_ids")

    def test_context_prompt_ids_prefers_stored_prompt_ids(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                raise AssertionError("stored prompt ids should avoid prompt reconstruction")

            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("stored prompt ids should avoid prompt reconstruction")

        prompt_ids, source = context_prompt_ids_for_condition(
            tokenizer=FakeTokenizer(),
            case={
                "problem": "2+2?",
                "context_records": {
                    "teacher_base": {
                        "condition": "teacher_base",
                        "prompt_token_ids": [7, 8, 9],
                    }
                },
            },
            condition="teacher_base",
            skeletons={},
        )

        self.assertEqual(prompt_ids, [7, 8, 9])
        self.assertEqual(source, "prompt_token_ids")

    def test_context_prompt_reconstruction_uses_record_student_thinking_mode(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2, 3] if "thinking=True" in text else [4, 5, 6]}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return f"thinking={enable_thinking}:{messages[0]['content']}"

        prompt_ids, source = context_prompt_ids_for_condition(
            tokenizer=FakeTokenizer(),
            case={
                "problem": "2+2?",
                "context_records": {
                    "student": {
                        "condition": "student",
                        "problem": "2+2?",
                        "enable_thinking": True,
                    }
                },
            },
            condition="student",
            skeletons={},
        )

        self.assertEqual(prompt_ids, [1, 2, 3])
        self.assertEqual(source, "reconstructed_prompt_text")

    def test_missing_teacher_context_reconstruction_keeps_teacher_thinking_enabled(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1] if "thinking=True" in text else [0]}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return f"thinking={enable_thinking}:{messages[0]['content']}"

        prompt_ids, source = context_prompt_ids_for_condition(
            tokenizer=FakeTokenizer(),
            case={
                "problem_id": 1,
                "problem": "2+2?",
                "solution": "2+2=4",
                "ground_truth": "4",
                "condition": "student",
                "enable_thinking": False,
                "context_records": {
                    "student": {
                        "condition": "student",
                        "problem": "2+2?",
                        "enable_thinking": False,
                    }
                },
            },
            condition="teacher_reference",
            skeletons={},
        )

        self.assertEqual(prompt_ids, [1])
        self.assertEqual(source, "reconstructed_prompt_text")

    def test_missing_teacher_context_does_not_reuse_student_prompt_ids(self):
        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=False):
                if "Reference Solution Begin" not in text:
                    raise AssertionError("teacher_reference prompt should be reconstructed")
                return {"input_ids": [99, 100]}

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                return f"thinking={enable_thinking}:{messages[0]['content']}"

        prompt_ids, source = context_prompt_ids_for_condition(
            tokenizer=FakeTokenizer(),
            case={
                "problem_id": 1,
                "problem": "2+2?",
                "solution": "2+2=4",
                "ground_truth": "4",
                "condition": "student",
                "enable_thinking": False,
                "prompt_token_ids": [7, 8, 9],
                "context_records": {
                    "student": {
                        "condition": "student",
                        "problem": "2+2?",
                        "enable_thinking": False,
                        "prompt_token_ids": [7, 8, 9],
                    }
                },
            },
            condition="teacher_reference",
            skeletons={},
        )

        self.assertEqual(prompt_ids, [99, 100])
        self.assertEqual(source, "reconstructed_prompt_text")

    def test_compare_contexts_uses_hf_logprob_rows(self):
        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {0: " wait", 1: " fraction"}[token_ids[0]]

        student_log_probs = [
            {0: -1.6094379124341003, 1: -0.2231435513142097},
            {0: -0.2231435513142097, 1: -1.6094379124341003},
        ]
        teacher_log_probs = [
            {0: -0.2231435513142097, 1: -1.6094379124341003},
            {0: -1.6094379124341003, 1: -0.2231435513142097},
        ]

        record = compare_contexts(
            case={"case_id": "1:0:teacher_base", "problem_id": 1, "target_condition": "teacher_base"},
            tokenizer=FakeTokenizer(),
            target_ids=[0, 1],
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            contrast="teacher_reference_vs_teacher_base",
            top_k=1,
            top_kl_positions=2,
            first_window_tokens=1,
        )

        self.assertEqual(record["logprob_backend"], "hf_causal_lm")
        self.assertEqual(record["target_token_source"], "target_tail_text")
        self.assertAlmostEqual(record["kl_per_token"][0], 0.8317766166719344)
        self.assertAlmostEqual(record["teacher_entropy_per_token"][0], 0.5004024235381879)
        self.assertEqual(record["top1_agreement"], 0.0)
        self.assertEqual(record["token_category_kl"]["style"]["num_tokens"], 1)
        self.assertAlmostEqual(record["token_category_kl"]["style"]["mean_kl"], 0.8317766166719344)
        self.assertEqual(record["token_category_kl"]["math"]["num_tokens"], 1)
        self.assertAlmostEqual(record["token_category_kl"]["math"]["mean_kl"], 0.8317766166719344)
        self.assertEqual(record["top_kl_positions"][0]["teacher_top_tokens"], [{"token": " wait", "prob": 0.8, "logprob": -0.2231435513142097}])
        self.assertEqual(record["top_kl_positions"][0]["base_top_tokens"], [{"token": " fraction", "prob": 0.8, "logprob": -0.2231435513142097}])

    def test_records_can_report_completion_token_id_source(self):
        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {0: "A", 1: "B"}[token_ids[0]]

        log_probs = [
            {0: -0.2231435513142097, 1: -1.6094379124341003},
            {0: -1.6094379124341003, 1: -0.2231435513142097},
        ]
        case = {"case_id": "1:0:teacher_base", "problem_id": 1, "target_condition": "teacher_base"}

        contrast_record = compare_contexts(
            case=case,
            tokenizer=FakeTokenizer(),
            target_ids=[0, 1],
            student_log_probs=log_probs,
            teacher_log_probs=log_probs,
            contrast="teacher_reference_vs_teacher_base",
            top_k=1,
            top_kl_positions=2,
            first_window_tokens=1,
            target_token_source="completion_token_ids",
        )
        entropy_record = build_rollout_entropy_record(
            case=case,
            tokenizer=FakeTokenizer(),
            target_ids=[0, 1],
            log_probs=log_probs,
            target_token_source="completion_token_ids",
        )

        self.assertEqual(contrast_record["target_token_source"], "completion_token_ids")
        self.assertEqual(entropy_record["target_token_source"], "completion_token_ids")

    def test_build_top_kl_positions_includes_token_context(self):
        positions = build_top_kl_positions(
            token_texts=["A", "B", "C"],
            kl_values=[0.2, 0.7, 0.5],
            delta_logp_values=[0.1, -0.2, 0.3],
            teacher_entropy_values=[1.0, 2.0, 3.0],
            student_entropy_values=[1.5, 1.0, 2.5],
            teacher_top_token_rows=[
                [{"token": "ta", "prob": 0.7, "logprob": -0.4}],
                [{"token": "tb", "prob": 0.8, "logprob": -0.2}],
                [{"token": "tc", "prob": 0.9, "logprob": -0.1}],
            ],
            base_top_token_rows=[
                [{"token": "ba", "prob": 0.6, "logprob": -0.5}],
                [{"token": "bb", "prob": 0.5, "logprob": -0.7}],
                [{"token": "bc", "prob": 0.4, "logprob": -0.9}],
            ],
            top_n=2,
        )

        self.assertEqual([row["position"] for row in positions], [1, 2])
        self.assertEqual(positions[0]["teacher_top_tokens"], [{"token": "tb", "prob": 0.8, "logprob": -0.2}])

    def test_summarize_logit_records_includes_entropy_metrics(self):
        summary = summarize_logit_records(
            [
                {
                    "record_type": "kl_contrast",
                    "contrast": "teacher_reference_vs_teacher_base",
                    "mean_kl": 0.1,
                    "top1_agreement": 0.2,
                    "topk_jaccard": 0.3,
                    "mean_delta_logp_target": 0.4,
                    "style_kl_share": 0.5,
                    "math_kl_share": 0.6,
                    "other_kl_share": 0.1,
                    "first_window_kl_share": 0.7,
                    "mean_teacher_entropy": 1.0,
                    "mean_student_entropy": 2.0,
                    "mean_delta_entropy": -1.0,
                    "token_category_kl": {
                        "style": {"num_tokens": 2, "sum_kl": 1.0, "mean_kl": 0.5},
                        "math": {"num_tokens": 1, "sum_kl": 0.6, "mean_kl": 0.6},
                        "other": {"num_tokens": 1, "sum_kl": 0.1, "mean_kl": 0.1},
                    },
                },
                {
                    "record_type": "kl_contrast",
                    "contrast": "teacher_reference_vs_teacher_base",
                    "mean_kl": 0.3,
                    "top1_agreement": 0.4,
                    "topk_jaccard": 0.5,
                    "mean_delta_logp_target": 0.6,
                    "style_kl_share": 0.7,
                    "math_kl_share": 0.8,
                    "other_kl_share": 0.3,
                    "first_window_kl_share": 0.9,
                    "mean_teacher_entropy": 3.0,
                    "mean_student_entropy": 4.0,
                    "mean_delta_entropy": -1.0,
                    "token_category_kl": {
                        "style": {"num_tokens": 1, "sum_kl": 0.2, "mean_kl": 0.2},
                        "math": {"num_tokens": 3, "sum_kl": 0.3, "mean_kl": 0.1},
                        "other": {"num_tokens": 2, "sum_kl": 0.4, "mean_kl": 0.2},
                    },
                },
                {"record_type": "rollout_entropy", "condition": "student", "mean_entropy": 1.0},
                {"record_type": "rollout_entropy", "condition": "student", "mean_entropy": 3.0},
                {"record_type": "rollout_entropy", "condition": "teacher_base", "mean_entropy": 4.0},
                {"record_type": "rollout_entropy", "condition": "teacher_skeleton", "mean_entropy": 5.0},
            ]
        )

        contrast = summary["contrasts"]["teacher_reference_vs_teacher_base"]
        self.assertEqual(contrast["mean_teacher_entropy"], 2.0)
        self.assertEqual(contrast["mean_student_entropy"], 3.0)
        self.assertEqual(contrast["mean_delta_entropy"], -1.0)
        self.assertAlmostEqual(contrast["mean_other_kl_share"], 0.2)
        self.assertAlmostEqual(contrast["token_category_kl"]["style"]["mean_kl"], 0.4)
        self.assertEqual(contrast["token_category_kl"]["style"]["num_tokens"], 3)
        self.assertAlmostEqual(contrast["token_category_kl"]["math"]["mean_kl"], 0.225)
        self.assertEqual(contrast["token_category_kl"]["math"]["num_tokens"], 4)
        self.assertAlmostEqual(contrast["token_category_kl"]["other"]["mean_kl"], 0.16666666666666666)
        self.assertEqual(contrast["token_category_kl"]["other"]["num_tokens"], 3)
        self.assertEqual(summary["rollout_entropy"]["student"]["mean_entropy"], 2.0)
        self.assertEqual(summary["rollout_entropy"]["teacher_base"]["mean_entropy"], 4.0)
        self.assertEqual(summary["rollout_entropy"]["teacher_skeleton"]["mean_entropy"], 5.0)


if __name__ == "__main__":
    unittest.main()
