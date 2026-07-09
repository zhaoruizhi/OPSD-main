import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "count_eval_response_tokens.py"


def load_module():
    spec = importlib.util.spec_from_file_location("count_eval_response_tokens", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class EvalResponseTokenCountTests(unittest.TestCase):
    def write_eval_file(self, eval_dir, filename, payload):
        path = eval_dir / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_nested_generations_average_counts_all_eval_responses(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            eval_dir = Path(temp_dir)
            path = self.write_eval_file(
                eval_dir,
                "aime25_reference_checkpoint_50.json",
                {
                    "dataset": "aime25",
                    "val_n": 2,
                    "average_at_n_pct": 43.1,
                    "pass_at_n_pct": 80.0,
                    "results": [
                        {
                            "problem_id": 1,
                            "generations": [
                                {"full_generation": "aa", "correct": True},
                                {"full_generation": "bbbb", "correct": False},
                            ],
                        },
                        {
                            "problem_id": 2,
                            "generations": [
                                {"full_generation": "c", "correct": False},
                                {"full_generation": "ddd", "correct": True},
                            ],
                        },
                    ],
                },
            )

            summary = module.summarize_eval_file(path, CharTokenizer())

        self.assertEqual(summary.dataset, "aime25")
        self.assertEqual(summary.condition, "reference")
        self.assertEqual(summary.checkpoint_step, 50)
        self.assertEqual(summary.response_count, 4)
        self.assertEqual(summary.problem_count, 2)
        self.assertEqual(summary.total_response_tokens, 10)
        self.assertEqual(summary.average_response_tokens, 2.5)
        self.assertEqual(summary.average_at_n_pct, 43.1)
        self.assertEqual(summary.pass_at_n_pct, 80.0)

    def test_top_level_full_generation_fallback_supports_legacy_eval_results(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            eval_dir = Path(temp_dir)
            path = self.write_eval_file(
                eval_dir,
                "hmmt25_skeleton_checkpoint_25.json",
                {
                    "dataset": "hmmt25",
                    "val_n": 1,
                    "results": [
                        {"problem_id": 1, "full_generation": "hello"},
                        {"problem_id": 2, "full_generation": "xy"},
                    ],
                },
            )

            summary = module.summarize_eval_file(path, CharTokenizer())

        self.assertEqual(summary.condition, "skeleton")
        self.assertEqual(summary.checkpoint_step, 25)
        self.assertEqual(summary.response_count, 2)
        self.assertEqual(summary.total_response_tokens, 7)
        self.assertEqual(summary.average_response_tokens, 3.5)

    def test_discover_eval_files_filters_dataset_condition_and_step(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            eval_dir = Path(temp_dir)
            wanted = self.write_eval_file(
                eval_dir,
                "aime25_skeleton_checkpoint_100.json",
                {"dataset": "aime25", "results": [{"full_generation": "ok"}]},
            )
            self.write_eval_file(
                eval_dir,
                "aime24_skeleton_checkpoint_100.json",
                {"dataset": "aime24", "results": [{"full_generation": "skip"}]},
            )
            self.write_eval_file(
                eval_dir,
                "aime25_reference_checkpoint_75.json",
                {"dataset": "aime25", "results": [{"full_generation": "skip"}]},
            )
            (eval_dir / "notes.txt").write_text("not json", encoding="utf-8")

            files = module.discover_eval_files(
                [eval_dir],
                datasets=["aime25"],
                conditions=["skeleton"],
                target_steps=[100],
            )

        self.assertEqual(files, [wanted])

    def test_rows_for_export_include_metrics_and_path(self):
        module = load_module()

        summary = module.EvalFileSummary(
            dataset="aime25",
            condition="reference",
            checkpoint_step=50,
            path=Path("/tmp/aime25_reference_checkpoint_50.json"),
            response_count=4,
            total_response_tokens=10,
            average_response_tokens=2.5,
            problem_count=2,
            val_n=2,
            average_at_n_pct=43.1,
            pass_at_n_pct=80.0,
            majority_vote_at_n_pct=60.0,
            format_rate=90.0,
        )

        rows = module.rows_for_export([summary])

        self.assertEqual(
            rows,
            [
                {
                    "dataset": "aime25",
                    "condition": "reference",
                    "checkpoint_step": 50,
                    "eval_file": "/tmp/aime25_reference_checkpoint_50.json",
                    "problem_count": 2,
                    "val_n": 2,
                    "response_count": 4,
                    "total_response_tokens": 10,
                    "average_response_tokens": 2.5,
                    "average_at_n_pct": 43.1,
                    "pass_at_n_pct": 80.0,
                    "majority_vote_at_n_pct": 60.0,
                    "format_rate": 90.0,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
