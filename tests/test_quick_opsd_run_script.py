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

    def test_student_teacher_category_kl_script_runs_four_rollouts_and_two_kl_probes(self):
        script = Path("scripts/run_student_teacher_category_kl.sh").read_text(encoding="utf-8")

        self.assertIn("SAMPLE_SIZE=10", script)
        self.assertIn("VAL_N=1", script)
        self.assertIn("STUDENT_TM=", script)
        self.assertIn('MAX_NEW_TOKENS="1024"', script)
        self.assertIn('MAX_NEW_TOKENS="16384"', script)
        self.assertIn("--gpu-ids)", script)
        self.assertNotIn("--condition student", script)
        self.assertIn('--skeleton-file "$OUT/skeletons.jsonl"', script)
        self.assertIn('rollout_shard${gpu}.jsonl', script)
        self.assertIn('rollouts.jsonl', script)
        self.assertIn('rollout_summary.json', script)
        self.assertIn("--student-enable-thinking", script)
        self.assertIn("--trajectory-condition teacher_base", script)
        self.assertIn("--baseline-condition teacher_base", script)
        self.assertIn('logit_probe_shard${gpu}.jsonl', script)
        self.assertIn('logit_summary.json', script)
        self.assertIn("--trajectory-condition student", script)
        self.assertIn("--baseline-condition student", script)
        self.assertIn("--teacher-condition teacher_reference", script)
        self.assertIn("--teacher-condition teacher_skeleton", script)
        self.assertEqual(script.count("--skip-rollout-entropy"), 1)
        self.assertGreaterEqual(script.count("--require-context-rollouts"), 2)
        self.assertGreaterEqual(script.count("CUDA_VISIBLE_DEVICES=$gpu"), 3)
        self.assertIn("student_teacher_category_kl_summary.json", script)
        self.assertLess(
            script.index("--trajectory-condition teacher_base"),
            script.index("--trajectory-condition student"),
        )
        self.assertLess(
            script.index("--trajectory-condition student"),
            script.index("run_teacher_spike_continuations.sh"),
        )

    def test_teacher_spike_runner_accepts_arbitrary_gpu_ids(self):
        script = Path("scripts/run_teacher_spike_continuations.sh").read_text(encoding="utf-8")

        self.assertIn('--gpu-ids)', script)
        self.assertIn('read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"', script)
        self.assertIn('CUDA_VISIBLE_DEVICES="$gpu"', script)
        self.assertIn('--max-new-tokens "$MAX_NEW_TOKENS"', script)
        self.assertIn('--shard-id "$shard_id"', script)
        self.assertIn('--num-shards "$NUM_SHARDS"', script)
        self.assertIn('--sort-key rank', script)
        self.assertIn('student_teacher_category_kl_remerged.jsonl', script)
        self.assertIn('teacher_spike_continuations.html', script)

    def test_category_kl_runner_integrates_teacher_spike_phase(self):
        script = Path("scripts/run_student_teacher_category_kl.sh").read_text(encoding="utf-8")

        self.assertIn("quick_teacher_base_kl_report.py", script)
        self.assertIn(
            "visualizations/teacher_base_kl_reference_vs_skeleton_report.html", script
        )
        self.assertIn(
            "visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv", script
        )
        self.assertIn(
            "visualizations/teacher_base_top_distribution_spikes.jsonl", script
        )
        self.assertGreater(
            script.index("quick_teacher_base_kl_report.py"),
            script.index('--summary-file "$OUT/logit_summary.json"'),
        )
        self.assertIn("run_teacher_spike_continuations.sh", script)
        self.assertIn("--teacher-continuation-top-n", script)
        self.assertIn("--teacher-continuation-max-new-tokens", script)
        self.assertIn("--skip-teacher-continuations", script)
        self.assertIn("quick_jsonl_merge.py", script)
        self.assertIn('--kl-file "$OUT/student_teacher_category_kl.jsonl"', script)
        self.assertIn('--student-rollout-file "$OUT/rollouts.jsonl"', script)


if __name__ == "__main__":
    unittest.main()
