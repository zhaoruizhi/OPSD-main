import hashlib
import tempfile
import unittest
from pathlib import Path


class ExperimentConfigTests(unittest.TestCase):
    def test_build_experiment_config_records_effective_settings_and_input_hashes(self):
        from eval.write_experiment_config import build_experiment_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "sample_indices.json"
            skeletons = root / "skeletons.jsonl"
            manifest.write_text('{"indices":[1,2]}\n', encoding="utf-8")
            skeletons.write_text('{"problem_id":1}\n', encoding="utf-8")

            payload = build_experiment_config(
                effective_config={
                    "experiment_profile": "legacy-20260629",
                    "val_n": 4,
                    "gpu_ids": [4, 5, 6, 7],
                    "target_token_source": "target_tail_text",
                },
                sample_indices_file=manifest,
                skeleton_file=skeletons,
                git_commit="abc123",
                git_dirty=True,
            )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["effective_config"]["experiment_profile"], "legacy-20260629")
        self.assertEqual(payload["effective_config"]["val_n"], 4)
        self.assertEqual(payload["git"], {"commit": "abc123", "dirty": True})
        self.assertEqual(
            payload["input_files"]["sample_indices"]["sha256"],
            hashlib.sha256(b'{"indices":[1,2]}\n').hexdigest(),
        )
        self.assertEqual(
            payload["input_files"]["skeletons"]["sha256"],
            hashlib.sha256(b'{"problem_id":1}\n').hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
