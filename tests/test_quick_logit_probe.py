import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.quick_opsd_common import summarize_logit_records
from eval.quick_logit_probe import (
    build_top_kl_positions,
    compare_contexts,
    completed_logit_record_keys,
    select_full_response_cases,
    shard_cases,
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

    def test_compare_contexts_uses_hf_logprob_rows(self):
        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return {0: "A", 1: "B"}[token_ids[0]]

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
        self.assertEqual(record["top_kl_positions"][0]["teacher_top_tokens"], [{"token": "A", "prob": 0.8, "logprob": -0.2231435513142097}])
        self.assertEqual(record["top_kl_positions"][0]["base_top_tokens"], [{"token": "B", "prob": 0.8, "logprob": -0.2231435513142097}])

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
                    "first_window_kl_share": 0.7,
                    "mean_teacher_entropy": 1.0,
                    "mean_student_entropy": 2.0,
                    "mean_delta_entropy": -1.0,
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
                    "first_window_kl_share": 0.9,
                    "mean_teacher_entropy": 3.0,
                    "mean_student_entropy": 4.0,
                    "mean_delta_entropy": -1.0,
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
        self.assertEqual(summary["rollout_entropy"]["student"]["mean_entropy"], 2.0)
        self.assertEqual(summary["rollout_entropy"]["teacher_base"]["mean_entropy"], 4.0)
        self.assertEqual(summary["rollout_entropy"]["teacher_skeleton"]["mean_entropy"], 5.0)


if __name__ == "__main__":
    unittest.main()
