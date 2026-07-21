import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DummyScriptArguments:
    teacher_context_mode: str = "skeleton"
    teacher_prompt_profile: str = "legacy-20260629"
    run_config: str = "legacy-train"


class DummyTrainingArguments:
    def to_dict(self):
        return {
            "output_dir": "/tmp/opsd/legacy-train",
            "max_completion_length": 1024,
            "temperature": 1.1,
        }


@dataclass
class DummyModelArguments:
    model_name_or_path: str = "/models/Qwen3-1.7B"
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj")


class TrainingExperimentManifestTests(unittest.TestCase):
    def test_manifest_records_effective_configs_git_runtime_command_and_skeleton_hash(self):
        from training_experiment_manifest import write_training_experiment_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)

            skeleton_file = repo / "skeletons.jsonl"
            skeleton_bytes = b'{"problem_id":0,"status":"ok"}\n'
            skeleton_file.write_bytes(skeleton_bytes)
            subprocess.run(["git", "add", "skeletons.jsonl"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

            output_file = root / "run" / "experiment_config.json"
            payload = write_training_experiment_manifest(
                output_file,
                script_args=DummyScriptArguments(),
                training_args=DummyTrainingArguments(),
                model_args=DummyModelArguments(),
                skeleton_file=skeleton_file,
                repo_root=repo,
                argv=["opsd_train.py", "--teacher_prompt_profile", "legacy-20260629"],
                environ={
                    "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                    "WORLD_SIZE": "4",
                    "LOCAL_RANK": "0",
                    "UNRELATED_SECRET": "must-not-be-recorded",
                },
            )

            disk_payload = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(payload, disk_payload)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(
            payload["effective_config"]["script_args"]["teacher_prompt_profile"],
            "legacy-20260629",
        )
        self.assertEqual(payload["effective_config"]["training_args"]["max_completion_length"], 1024)
        self.assertEqual(payload["effective_config"]["model_args"]["lora_target_modules"], ["q_proj", "k_proj"])
        self.assertEqual(
            payload["command"]["argv"],
            ["opsd_train.py", "--teacher_prompt_profile", "legacy-20260629"],
        )
        self.assertEqual(
            payload["runtime"]["environment"],
            {"CUDA_VISIBLE_DEVICES": "4,5,6,7", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        )
        self.assertNotIn("UNRELATED_SECRET", json.dumps(payload))
        self.assertEqual(payload["git"], {"commit": commit, "dirty": True})
        self.assertEqual(
            payload["input_files"]["skeletons"]["sha256"],
            hashlib.sha256(skeleton_bytes).hexdigest(),
        )
        self.assertEqual(
            payload["input_files"]["skeletons"]["path"],
            str(skeleton_file.resolve()),
        )
        self.assertIn("created_at_utc", payload)


if __name__ == "__main__":
    unittest.main()
