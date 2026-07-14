import sys
import types
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory


class FakeTokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.pad_token_id = 0
        self.chat_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return f"<chat>{messages[0]['content']}<gen>"

    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        encoded = []
        for text in texts:
            token_ids = list(range(1, len(str(text).split()) + 1)) or [1]
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            encoded.append(token_ids)

        target_length = None
        if padding == "max_length":
            target_length = int(max_length)
        elif padding in {True, "longest"}:
            target_length = max(len(ids) for ids in encoded)

        attention_masks = []
        if target_length is not None:
            padded = []
            for ids in encoded:
                pad_count = max(0, target_length - len(ids))
                padded_ids = ids + [self.pad_token_id] * pad_count
                padded.append(padded_ids)
                attention_masks.append([1] * len(ids) + [0] * pad_count)
            encoded = padded
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]

        if return_tensors == "pt":
            return {"input_ids": encoded, "attention_mask": attention_masks}
        return {"input_ids": encoded, "attention_mask": attention_masks}


class OpsdSkeletonTrainingTests(unittest.TestCase):
    def setUp(self):
        self._installed_fake_torch = False
        if "torch" not in sys.modules:
            fake_torch = types.SimpleNamespace(long="long")
            fake_torch.tensor = lambda value, dtype=None: value
            sys.modules["torch"] = fake_torch
            self._installed_fake_torch = True

    def tearDown(self):
        if self._installed_fake_torch:
            sys.modules.pop("torch", None)

    def test_attach_skeletons_to_training_rows_requires_complete_file_by_default(self):
        from opsd_skeleton import attach_skeletons_to_training_rows, write_skeleton_training_rows

        rows = [
            {"problem": "p0", "solution": "full reference 0"},
            {"problem": "p1", "solution": "full reference 1"},
        ]

        with TemporaryDirectory() as temp_dir:
            skeleton_file = Path(temp_dir) / "skeletons.jsonl"
            write_skeleton_training_rows(
                skeleton_file,
                [
                    {
                        "problem_id": 0,
                        "ground_truth": "4",
                        "status": "ok",
                        "skeleton": {
                            "final_answer": "4",
                            "key_objects": [],
                            "subgoals": ["establish p0"],
                            "critical_intermediates": [],
                            "theorem_tags": [],
                            "checks": [],
                        },
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "missing skeletons.*1"):
                attach_skeletons_to_training_rows(rows, skeleton_file, subset_policy="error")

    def test_attach_skeletons_to_training_rows_filters_missing_rows_when_requested(self):
        from opsd_skeleton import attach_skeletons_to_training_rows, write_skeleton_training_rows

        rows = [
            {"problem": "p0", "solution": "full reference 0"},
            {"problem": "p1", "solution": "full reference 1"},
        ]

        with TemporaryDirectory() as temp_dir:
            skeleton_file = Path(temp_dir) / "skeletons.jsonl"
            write_skeleton_training_rows(
                skeleton_file,
                [
                    {
                        "problem_id": 1,
                        "ground_truth": "5",
                        "status": "ok",
                        "skeleton": {
                            "final_answer": "5",
                            "key_objects": [],
                            "subgoals": ["establish p1"],
                            "critical_intermediates": [],
                            "theorem_tags": [],
                            "checks": [],
                        },
                    }
                ],
            )

            prepared = attach_skeletons_to_training_rows(rows, skeleton_file, subset_policy="filter")

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["problem"], "p1")
        self.assertEqual(prepared[0]["solution"], "full reference 1")
        self.assertEqual(prepared[0]["ground_truth"], "5")
        self.assertEqual(prepared[0]["semantic_skeleton"]["subgoals"], ["establish p1"])

    def test_collator_reference_mode_preserves_existing_teacher_prompt(self):
        from data_collator import SelfDistillationDataCollator

        tokenizer = FakeTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer=tokenizer,
            teacher_context_mode="reference",
            reason_first=False,
        )

        collator([{"problem": "Compute 2+2.", "solution": "Full reference solution."}])

        teacher_content = tokenizer.chat_calls[1]["messages"][0]["content"]
        self.assertIn("Full reference solution.", teacher_content)
        self.assertIn("=== Reference Solution Begin ===", teacher_content)
        self.assertNotIn("Final answer:", teacher_content)
        self.assertNotIn("Interpret the fields as follows:", teacher_content)

    def test_collator_skeleton_mode_uses_skeleton_without_full_reference_or_answer_in_block(self):
        from data_collator import SelfDistillationDataCollator

        tokenizer = FakeTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer=tokenizer,
            teacher_context_mode="skeleton",
            reason_first=False,
        )

        collator(
            [
                {
                    "problem": "Compute 2+2.",
                    "solution": "Full reference solution must not appear.",
                    "ground_truth": "4",
                    "semantic_skeleton": {
                        "final_answer": "4",
                        "key_objects": [],
                        "subgoals": ["evaluate the sum"],
                        "critical_intermediates": ["2+2=4"],
                        "theorem_tags": [],
                        "checks": [],
                    },
                }
            ]
        )

        teacher_content = tokenizer.chat_calls[1]["messages"][0]["content"]
        skeleton_block = teacher_content.split("=== Reference Solution Begin ===\n", 1)[1].split(
            "\n=== Reference Solution End ===", 1
        )[0]
        self.assertIn("Final answer: 4", teacher_content)
        self.assertEqual(teacher_content.count("Final answer: 4"), 1)
        self.assertLess(teacher_content.index("Final answer: 4"), teacher_content.index("Reference Solution Begin"))
        self.assertIn(
            "Here is a reference solution to this problem:\n\n=== Reference Solution Begin ===",
            teacher_content,
        )
        self.assertIn('"subgoals"', skeleton_block)
        self.assertIn('"check"', skeleton_block)
        self.assertNotIn('"checks"', skeleton_block)
        self.assertIn("evaluate the sum", skeleton_block)
        self.assertNotIn("final_answer", skeleton_block)
        self.assertNotIn("Full reference solution must not appear.", teacher_content)

        field_guidance = (
            'Interpret the fields as follows:\n'
            '- "key_objects" records potentially important mathematical objects and constraints.\n'
            '- "subgoals" records possible mathematical objectives.\n'
            '- "critical_intermediates" records potentially useful mathematical checkpoints. '
            'They are not mandatory generated sentences and do not imply that the reference path is the only valid path.\n'
            '- "theorem_tags" records optional and non-exclusive methods. Do not force a listed theorem when another valid approach is more natural.\n'
            '- "check" records validity conditions or possible failure modes. Apply a check only when it is relevant to the reasoning being used.'
        )
        self.assertIn(f"=== Reference Solution End ===\n\n{field_guidance}\n\nAfter reading", teacher_content)

    def test_collator_skeleton_mode_accepts_serialized_skeleton(self):
        from data_collator import SelfDistillationDataCollator

        tokenizer = FakeTokenizer()
        collator = SelfDistillationDataCollator(
            tokenizer=tokenizer,
            teacher_context_mode="skeleton",
            reason_first=False,
        )

        collator(
            [
                {
                    "problem": "Compute 2+2.",
                    "solution": "Full reference solution must not appear.",
                    "ground_truth": "4",
                    "semantic_skeleton": json.dumps(
                        {
                            "final_answer": "4",
                            "key_objects": [{"name": "the expression 2+2"}],
                            "subgoals": ["evaluate the sum"],
                            "critical_intermediates": ["2+2=4"],
                            "theorem_tags": [],
                            "checks": [],
                        }
                    ),
                }
            ]
        )

        teacher_content = tokenizer.chat_calls[1]["messages"][0]["content"]
        self.assertIn("Final answer: 4", teacher_content)
        self.assertIn('"key_objects"', teacher_content)
        self.assertIn('"name": "the expression 2+2"', teacher_content)
        self.assertNotIn("Full reference solution must not appear.", teacher_content)

    def test_skeleton_run_script_uses_skeleton_mode_and_distinct_run_config(self):
        script = Path("scripts/run_opsd_1b_skeleton.sh").read_text(encoding="utf-8")

        self.assertIn("SKELETON_FILE=", script)
        self.assertIn('TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"', script)
        self.assertIn('NUM_PROCESSES="${NUM_PROCESSES:-4}"', script)
        self.assertIn('MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" accelerate launch', script)
        self.assertIn('--num_processes "$NUM_PROCESSES"', script)
        self.assertIn('--main_process_port "$MAIN_PROCESS_PORT"', script)
        self.assertIn("--run_config qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005", script)
        self.assertIn("--teacher_context_mode skeleton", script)
        self.assertIn('--skeleton_file "$SKELETON_FILE"', script)
        self.assertIn("--skeleton_subset_policy error", script)
        self.assertIn("--report_to wandb", script)

    def test_reference_run_script_supports_gpu_selection_and_wandb(self):
        script = Path("scripts/run_opsd_1b.sh").read_text(encoding="utf-8")

        self.assertIn('TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"', script)
        self.assertIn('NUM_PROCESSES="${NUM_PROCESSES:-4}"', script)
        self.assertIn('MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" accelerate launch', script)
        self.assertIn('--num_processes "$NUM_PROCESSES"', script)
        self.assertIn('--main_process_port "$MAIN_PROCESS_PORT"', script)
        self.assertIn("--run_config qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005", script)
        self.assertIn("--report_to wandb", script)
        self.assertIn("--wandb_project OPSD", script)

    def test_training_doc_describes_full_reference_vs_skeleton_workflow(self):
        doc = Path("docs/opsd_skeleton_training_zh.md").read_text(encoding="utf-8")

        self.assertNotIn("Smoke run", doc)
        self.assertIn("## 全量训练对比：reference baseline vs skeleton", doc)
        self.assertIn("wandb login", doc)
        self.assertIn("TRAIN_GPU_IDS=0,1,2,3", doc)
        self.assertIn("bash scripts/run_opsd_1b.sh", doc)
        self.assertIn("bash scripts/run_opsd_1b_skeleton.sh", doc)
        self.assertIn("loss", doc)
        self.assertIn("on_policy_loss", doc)
        self.assertIn("grad_norm", doc)
        self.assertIn("learning_rate", doc)


if __name__ == "__main__":
    unittest.main()
