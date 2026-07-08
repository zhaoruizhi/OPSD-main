import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "count_generation_tokens.py"


def load_module():
    spec = importlib.util.spec_from_file_location("count_generation_tokens", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class GenerationTokenCountTests(unittest.TestCase):
    def write_generation_file(self, generations_dir, step, completions):
        payload = {
            "step": step,
            "num_samples": len(completions),
            "generations": [
                {"step": step - 1, "prompt": f"prompt-{idx}", "completion": completion}
                for idx, completion in enumerate(completions)
            ],
        }
        path = generations_dir / f"generations_step_{step}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_cumulative_counts_include_all_generation_files_up_to_target_step(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            generations_dir = run_dir / "generations"
            generations_dir.mkdir(parents=True)
            self.write_generation_file(generations_dir, 5, ["aa", "bbb"])
            self.write_generation_file(generations_dir, 10, ["c"])
            self.write_generation_file(generations_dir, 25, ["dddd"])
            self.write_generation_file(generations_dir, 50, ["ee"])

            summary = module.summarize_run(
                label="reference",
                run_path=run_dir,
                tokenizer=CharTokenizer(),
                target_steps=[25, 50],
            )

        self.assertEqual(summary.label, "reference")
        self.assertEqual(summary.tokens_by_target, {25: 10, 50: 12})
        self.assertEqual(summary.samples_by_target, {25: 4, 50: 5})
        self.assertEqual(summary.files_by_target, {25: 3, 50: 4})

    def test_run_spec_can_label_generation_directory_directly(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            generations_dir = Path(temp_dir) / "generations"
            generations_dir.mkdir()
            self.write_generation_file(generations_dir, 5, ["abc"])

            label, path = module.parse_run_spec(f"skeleton={generations_dir}")
            summary = module.summarize_run(label, path, CharTokenizer(), [5])

        self.assertEqual(label, "skeleton")
        self.assertEqual(summary.tokens_by_target[5], 3)

    def test_missing_completion_field_fails_loudly(self):
        module = load_module()

        with TemporaryDirectory() as temp_dir:
            generations_dir = Path(temp_dir) / "generations"
            generations_dir.mkdir()
            bad_payload = {
                "step": 5,
                "num_samples": 1,
                "generations": [{"step": 4, "prompt": "prompt only"}],
            }
            (generations_dir / "generations_step_5.json").write_text(
                json.dumps(bad_payload), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "missing string field 'completion'"):
                module.summarize_run("bad", generations_dir, CharTokenizer(), [5])


if __name__ == "__main__":
    unittest.main()
