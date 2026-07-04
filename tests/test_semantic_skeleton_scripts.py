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

    def test_parse_skeleton_response_repairs_latex_style_backslashes(self):
        from eval.generate_semantic_skeletons import parse_skeleton_response

        skeleton = parse_skeleton_response(
            r'''{
              "final_answer": "\frac{1}{2}",
              "key_objects": [
                {
                  "name": "x",
                  "constraints": ["x \geq 0", "\$2.75"]
                }
              ],
              "subgoals": ["Use \left(\frac{x}{2}\right) and \pmod{5}"],
              "critical_intermediates": ["\sqrt{x}"],
              "theorem_tags": [],
              "checks": []
            }'''
        )

        self.assertEqual(skeleton["final_answer"], r"\frac{1}{2}")
        self.assertEqual(skeleton["key_objects"][0]["constraints"], [r"x \geq 0", r"\$2.75"])
        self.assertEqual(skeleton["subgoals"], [r"Use \left(\frac{x}{2}\right) and \pmod{5}"])
        self.assertEqual(skeleton["critical_intermediates"], [r"\sqrt{x}"])

    def test_parse_skeleton_response_accepts_json_inside_code_fence(self):
        from eval.generate_semantic_skeletons import parse_skeleton_response

        skeleton = parse_skeleton_response(
            """```json
{"final_answer":"4","key_objects":[],"subgoals":[],"critical_intermediates":[],"theorem_tags":[],"checks":[]}
```"""
        )

        self.assertEqual(skeleton["final_answer"], "4")

    def test_generate_skeleton_record_can_use_injected_local_completion(self):
        from eval.generate_semantic_skeletons import generate_skeleton_record

        calls = []

        def local_completion(*, answer, reference_solution):
            calls.append({"answer": answer, "reference_solution": reference_solution})
            return (
                '{"final_answer":"4","key_objects":[],"subgoals":["establish the sum"],'
                '"critical_intermediates":["2+2=4"],"theorem_tags":[],"checks":[]}'
            )

        record = generate_skeleton_record(
            problem_id=11,
            example={"answer": "4", "solution": "Compute 2+2 and conclude."},
            api_key=None,
            base_url=None,
            model="/data0/shared/Qwen3-1.7B",
            temperature=0.0,
            max_tokens=128,
            timeout=1.0,
            max_retries=0,
            skeleton_backend="vllm",
            completion_fn=local_completion,
        )

        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["skeleton_backend"], "vllm")
        self.assertEqual(record["model"], "/data0/shared/Qwen3-1.7B")
        self.assertEqual(record["skeleton"]["subgoals"], ["establish the sum"])
        self.assertEqual(calls, [{"answer": "4", "reference_solution": "Compute 2+2 and conclude."}])

    def test_generate_skeleton_record_converts_network_timeout_to_error_record(self):
        from eval.generate_semantic_skeletons import generate_skeleton_record

        calls = 0

        def timeout_completion(*, answer, reference_solution):
            nonlocal calls
            calls += 1
            raise TimeoutError("API read timed out")

        record = generate_skeleton_record(
            problem_id=12,
            example={"answer": "4", "solution": "Compute 2+2 and conclude."},
            api_key=None,
            base_url=None,
            model="deepseek-v4-pro",
            temperature=0.0,
            max_tokens=128,
            timeout=1.0,
            max_retries=1,
            skeleton_backend="api",
            completion_fn=timeout_completion,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["problem_id"], 12)
        self.assertIn("API read timed out", record["error"])

    def test_generate_skeleton_records_can_parallelize_api_calls_and_keep_order(self):
        from eval.generate_semantic_skeletons import generate_skeleton_records
        import threading
        import time

        thread_names = []

        def local_completion(*, answer, reference_solution):
            thread_names.append(threading.current_thread().name)
            time.sleep(0.05)
            return (
                '{"final_answer":"4","key_objects":[],"subgoals":["establish the sum"],'
                '"critical_intermediates":["2+2=4"],"theorem_tags":[],"checks":[]}'
            )

        records = generate_skeleton_records(
            indices=[0, 1, 2],
            rows=[
                {"answer": "4", "solution": "Compute 2+2 and conclude."},
                {"answer": "4", "solution": "Compute 2+2 and conclude."},
                {"answer": "4", "solution": "Compute 2+2 and conclude."},
            ],
            api_key=None,
            base_url=None,
            model="/data0/shared/Qwen3-1.7B",
            temperature=0.0,
            max_tokens=128,
            timeout=1.0,
            max_retries=0,
            skeleton_backend="api",
            completion_fn=local_completion,
            api_concurrency=2,
        )

        self.assertEqual([record["problem_id"] for record in records], [0, 1, 2])
        self.assertEqual(len(thread_names), 3)
        self.assertGreaterEqual(len(set(thread_names)), 2)

    def test_write_jsonl_stream_writes_incrementally(self):
        from eval.generate_semantic_skeletons import write_jsonl_stream
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            records = (
                {"problem_id": index, "status": "ok"}
                for index in range(3)
            )

            write_jsonl_stream(output_path, records, flush_every=2)

            self.assertTrue(output_path.exists())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{"problem_id": 0, "status": "ok"}\n'
                '{"problem_id": 1, "status": "ok"}\n'
                '{"problem_id": 2, "status": "ok"}\n',
            )

    def test_resume_helpers_skip_existing_problem_ids(self):
        from eval.generate_semantic_skeletons import filter_missing_indices, load_existing_problem_ids
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            output_path.write_text(
                '{"problem_id": 0, "status": "ok"}\n'
                '{"problem_id": 2, "status": "ok"}\n'
                '{"problem_id": "bad"}\n'
                'not-json\n',
                encoding="utf-8",
            )

            existing_ids = load_existing_problem_ids(output_path)
            self.assertEqual(existing_ids, {0, 2})
            self.assertEqual(filter_missing_indices([0, 1, 2, 3], existing_ids), [1, 3])

    def test_resume_helpers_treat_existing_error_records_as_visited(self):
        from eval.generate_semantic_skeletons import filter_missing_indices, load_existing_problem_ids
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            output_path.write_text(
                '{"problem_id": 0, "status": "ok"}\n'
                '{"problem_id": 1, "status": "error", "error": "API read timed out"}\n',
                encoding="utf-8",
            )

            existing_ids = load_existing_problem_ids(output_path)
            self.assertEqual(existing_ids, {0, 1})
            self.assertEqual(filter_missing_indices([0, 1, 2], existing_ids), [2])

    def test_resume_helpers_continue_after_existing_error_holes(self):
        from eval.generate_semantic_skeletons import filter_missing_indices, load_existing_problem_ids
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            output_path.write_text(
                '{"problem_id": 0, "status": "ok"}\n'
                '{"problem_id": 1, "status": "error", "error": "API read timed out"}\n'
                '{"problem_id": 2, "status": "ok"}\n',
                encoding="utf-8",
            )

            existing_ids = load_existing_problem_ids(output_path)
            self.assertEqual(filter_missing_indices([0, 1, 2, 3, 4], existing_ids), [3, 4])

    def test_existing_summary_keeps_only_first_ok_record(self):
        from eval.generate_semantic_skeletons import load_existing_skeleton_summary
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            output_path.write_text(
                '{"problem_id": 0, "status": "error", "error": "bad"}\n'
                '{"problem_id": 1, "status": "ok", "skeleton": {"final_answer": "1"}}\n'
                '{"problem_id": 1, "status": "ok", "skeleton": {"final_answer": "duplicate"}}\n'
                '{"problem_id": 2, "status": "error", "error": "timeout"}\n',
                encoding="utf-8",
            )

            summary = load_existing_skeleton_summary(output_path)

            self.assertEqual(summary.seen_problem_ids, {0, 1, 2})
            self.assertEqual(set(summary.ok_records), {1})
            self.assertEqual(summary.ok_records[1]["skeleton"]["final_answer"], "1")
            self.assertEqual(summary.error_count, 2)
            self.assertEqual(summary.duplicate_count, 1)

    def test_filter_pending_indices_continues_forward_before_repairing_old_errors(self):
        from eval.generate_semantic_skeletons import filter_pending_indices

        self.assertEqual(
            filter_pending_indices(
                [0, 1, 2, 3, 4, 5],
                existing_ok_problem_ids={1, 2},
                existing_seen_problem_ids={0, 1, 2, 3},
            ),
            [4, 5, 0, 3],
        )

    def test_rewrite_clean_output_file_drops_errors_and_duplicate_ok_records(self):
        from eval.generate_semantic_skeletons import load_existing_skeleton_summary, rewrite_clean_output_file
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "skeletons.jsonl"
            output_path.write_text(
                '{"problem_id": 0, "status": "error", "error": "bad"}\n'
                '{"problem_id": 1, "status": "ok", "skeleton": {"final_answer": "1"}}\n'
                '{"problem_id": 1, "status": "ok", "skeleton": {"final_answer": "duplicate"}}\n'
                '{"problem_id": 2, "status": "ok", "skeleton": {"final_answer": "2"}}\n',
                encoding="utf-8",
            )

            summary = load_existing_skeleton_summary(output_path)
            rewrite_clean_output_file(output_path, summary.ok_records, [0, 1, 2])

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '{"problem_id": 1, "status": "ok", "skeleton": {"final_answer": "1"}}\n'
                '{"problem_id": 2, "status": "ok", "skeleton": {"final_answer": "2"}}\n',
            )

    def test_iter_skeleton_records_retries_until_ok_without_yielding_error(self):
        from eval.generate_semantic_skeletons import iter_skeleton_records

        calls = 0

        def flaky_completion(*, answer, reference_solution):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("API read timed out")
            return (
                '{"final_answer":"4","key_objects":[],"subgoals":["establish the sum"],'
                '"critical_intermediates":["2+2=4"],"theorem_tags":[],"checks":[]}'
            )

        records = list(
            iter_skeleton_records(
                indices=[0],
                rows=[{"answer": "4", "solution": "Compute 2+2 and conclude."}],
                api_key=None,
                base_url=None,
                model="deepseek-v4-pro",
                temperature=0.0,
                max_tokens=128,
                timeout=1.0,
                max_retries=0,
                skeleton_backend="api",
                completion_fn=flaky_completion,
                api_concurrency=1,
                retry_until_ok=True,
                retry_delay=0.0,
                max_retry_delay=0.0,
            )
        )

        self.assertEqual(calls, 2)
        self.assertEqual([record["status"] for record in records], ["ok"])

    def test_generate_skeleton_args_allow_full_split_without_manifest(self):
        from eval.generate_semantic_skeletons import parse_args

        args = parse_args(
            [
                "--output-file",
                "skeletons.jsonl",
                "--skeleton-backend",
                "api",
            ]
        )

        self.assertIsNone(args.sample_indices_file)

    def test_resolve_generation_indices_defaults_to_full_dataset_without_manifest(self):
        from eval.generate_semantic_skeletons import resolve_generation_indices

        self.assertEqual(
            resolve_generation_indices(row_count=4, sample_indices_file=None),
            [0, 1, 2, 3],
        )

    def test_render_skeleton_compiler_prompt_uses_system_user_messages_and_thinking_flag(self):
        from eval.generate_semantic_skeletons import SYSTEM_PROMPT, render_skeleton_compiler_prompt

        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append({"messages": messages, "kwargs": kwargs})
                return "rendered prompt"

        tokenizer = FakeTokenizer()

        prompt = render_skeleton_compiler_prompt(
            tokenizer,
            answer="4",
            reference_solution="Compute 2+2.",
            enable_thinking=False,
        )

        self.assertEqual(prompt, "rendered prompt")
        self.assertEqual(
            tokenizer.calls[0]["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "ANSWER:\n4\n\nREFERENCE_SOLUTION:\nCompute 2+2."},
            ],
        )
        self.assertEqual(
            tokenizer.calls[0]["kwargs"],
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        )

    def test_run_script_exposes_vllm_skeleton_backend_controls(self):
        from pathlib import Path

        script = Path("scripts/run_semantic_skeleton_ablation.sh").read_text(encoding="utf-8")

        self.assertIn('SKELETON_BACKEND="${SKELETON_BACKEND:-api}"', script)
        self.assertIn("--skeleton-backend)", script)
        self.assertIn("--skeleton-model)", script)
        self.assertIn("--skeleton-gpus)", script)
        self.assertIn('--skeleton-backend "$SKELETON_BACKEND"', script)
        self.assertIn('--skeleton-model "$SKELETON_MODEL_FOR_RUN"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES="$SKELETON_GPUS"', script)

    def test_reference_prompt_keeps_solution_and_exposes_ground_truth(self):
        from eval.quick_opsd_common import build_reference_user_message

        prompt = build_reference_user_message(
            problem="Compute 2+2.",
            solution="A reference solution says 2+2=4.",
            ground_truth="4",
        )

        self.assertIn("A reference solution says 2+2=4.", prompt)
        self.assertIn("Final answer: 4", prompt)
        self.assertLess(prompt.index("Final answer: 4"), prompt.index("Reference Solution Begin"))
        self.assertIn("Please reason step by step, and put your final answer within \\boxed{}.", prompt)

    def test_skeleton_prompt_uses_reference_boundaries_and_exposes_ground_truth_once(self):
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
            ground_truth="4",
        )

        self.assertIn("Here is a reference solution to this problem:", prompt)
        self.assertIn("=== Reference Solution Begin ===", prompt)
        self.assertIn("=== Reference Solution End ===", prompt)
        self.assertNotIn("Semantic Skeleton Begin", prompt)
        self.assertNotIn("Semantic Skeleton End", prompt)
        self.assertNotIn("semantic skeleton", prompt.lower())
        skeleton_block = prompt.split("=== Reference Solution Begin ===\n", 1)[1].split(
            "\n=== Reference Solution End ===", 1
        )[0]
        self.assertNotIn("final_answer", skeleton_block)
        self.assertIn("Final answer: 4", prompt)
        self.assertLess(prompt.index("Final answer: 4"), prompt.index("Reference Solution Begin"))
        self.assertIn("establish the sum", skeleton_block)


if __name__ == "__main__":
    unittest.main()
