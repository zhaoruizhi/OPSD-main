from pathlib import Path
import unittest


class QuickOpsdRunScriptTests(unittest.TestCase):
    def test_semantic_skeleton_run_script_orchestrates_ablation_phases(self):
        script = Path("scripts/run_semantic_skeleton_ablation.sh").read_text(encoding="utf-8")

        self.assertIn("prepare_sample_manifest.py", script)
        self.assertIn("generate_semantic_skeletons.py", script)
        self.assertIn("quick_rollout_openthoughts.py", script)
        self.assertIn("quick_logit_probe.py", script)
        self.assertIn("SAMPLE_SIZE=128", script)
        self.assertIn("VAL_N=4", script)
        self.assertIn("--sample-indices-file", script)
        self.assertIn("--skeleton-file", script)
        self.assertIn("rollout_summary.json", script)
        self.assertIn("logit_summary.json", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=$gpu", script)
        self.assertIn('GPU_IDS="${GPU_IDS:-4 5 6 7}"', script)
        self.assertIn("--gpus)", script)
        self.assertIn("read -r -a GPU_ID_ARRAY", script)
        self.assertIn('for gpu_index in "${!GPU_ID_ARRAY[@]}"', script)
        self.assertIn('gpu="${GPU_ID_ARRAY[$gpu_index]}"', script)
        self.assertIn('shard_id="$gpu_index"', script)
        self.assertIn('NUM_SHARDS="${#GPU_ID_ARRAY[@]}"', script)
        self.assertIn("PROBE_TOKENS=0", script)
        self.assertIn("TRAJECTORY_SAMPLE_INDEX=0", script)
        self.assertIn("SKIP_ROLLOUT_ENTROPY=0", script)
        self.assertIn("--trajectory-sample-index)", script)
        self.assertIn("full-response logit distribution probe", script)
        self.assertIn('--rollout-file "$OUT/rollouts.jsonl"', script)
        self.assertIn("--trajectory-condition teacher_base", script)
        self.assertIn('--trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX"', script)
        self.assertIn("--skip-rollout-entropy", script)
        self.assertIn('--shard-id "$shard_id"', script)
        self.assertIn('--num-shards "$NUM_SHARDS"', script)
        self.assertIn('logit_probe_shard${gpu}.jsonl', script)
        self.assertIn('--summarize-only', script)
        self.assertNotIn("SCORE_BATCH_SIZE", script)
        self.assertNotIn("--score-batch-size", script)
        self.assertNotIn("quick_prefix_intervention.py", script)
        self.assertNotIn("prefix_summary.json", script)

    def test_first_error_run_script_exposes_phase_b_resume_control(self):
        script = Path("scripts/run_first_error_ablation.sh").read_text(encoding="utf-8")

        self.assertIn('FIRST_ERROR_RESUME_ARGS=("--resume")', script)
        self.assertIn("--first-error-resume)", script)
        self.assertIn("--no-first-error-resume)", script)
        self.assertIn('FIRST_ERROR_RESUME_ARGS=("--no-resume")', script)
        self.assertIn('"${FIRST_ERROR_RESUME_ARGS[@]}"', script)
        self.assertLess(
            script.index("--output-file \"$OUT/first_error.jsonl\""),
            script.index('"${FIRST_ERROR_RESUME_ARGS[@]}"'),
        )


if __name__ == "__main__":
    unittest.main()
