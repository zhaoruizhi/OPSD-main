# Semantic Skeleton Ablation 实验说明

## 实验目标

这个实验比较完整 reference solution 和抽象 semantic skeleton 对 thinking teacher 的影响。主实验包含四路：

- `student`: non-thinking，prompt 只包含 problem。
- `teacher_base`: thinking，prompt 只包含 problem。
- `teacher_reference`: thinking，prompt 包含完整 reference solution 和 final answer。
- `teacher_skeleton`: thinking，prompt 包含 style-neutral semantic skeleton JSON 和 final answer。

主指标：

- 四路 rollout 的 `pass@4`。
- 四路 rollout 的 self-entropy。
- 沿 `teacher_base` rollout 轨迹计算 `teacher_reference_vs_teacher_base` 和 `teacher_skeleton_vs_teacher_base` KL。
- 高 KL 位置保存 `teacher_top_tokens` 和 `base_top_tokens`，用于检查 teacher 相对 base 更偏向哪些 token。

## 数据流

主入口：

```bash
scripts/run_semantic_skeleton_ablation.sh
```

阶段：

1. Phase 0: `eval/prepare_sample_manifest.py`
   - 输出 `$OUT/sample_indices.json`。

2. Phase 1: `eval/generate_semantic_skeletons.py`
   - 输出 `$OUT/skeletons.jsonl`。
   - skeleton compiler 只看到 dataset 的 answer 和 reference solution，不看到 problem statement。

3. Phase 2: `eval/quick_rollout_openthoughts.py`
   - 用 vLLM 生成四路 rollout。
   - 输出 `$OUT/rollout_shard*.jsonl`、`$OUT/rollouts.jsonl`、`$OUT/rollout_summary.json`。
   - rollout 记录不保存原始 token ids。

4. Phase 3: `eval/quick_logit_probe.py`
   - 使用 HuggingFace `AutoModelForCausalLM` dense forward 计算 full-response KL 和 entropy。
   - target 轨迹来自 `full_generation` / `target_tail_text` 文本重新 tokenize。
   - 输出 `$OUT/logit_probe.jsonl`、`$OUT/logit_summary.json`。

## Prompt 是怎么使用 skeleton 的

`teacher_reference` 使用完整 reference solution prompt。`teacher_skeleton` 使用 semantic skeleton 专属 prompt，不把 skeleton 称为 reference solution：

```text
Problem: {problem}

Here is a style-neutral semantic skeleton extracted from a reference solution:
=== Semantic Skeleton Begin ===
{skeleton_json_without_final_answer}
=== Semantic Skeleton End ===

Final answer: {answer}

Use the semantic skeleton above only as privileged mathematical guidance. Do not copy or paraphrase any reference wording; reason in your own words, fill in the missing derivation, and derive the same final answer independently.

Please reason step by step, and put your final answer within \boxed{}.
```

注意：

- `teacher_skeleton` prompt 不包含完整 reference solution。
- skeleton JSON 不包含 `final_answer` 字段；final answer 单独放在 `Final answer: ...` 行。
- `teacher_base` 只包含 problem，不包含 privileged info。

## 正式实验怎么跑

准备 API 环境变量：

```bash
export SKELETON_API_KEY="你的_API_KEY"
export SKELETON_BASE_URL="https://你的-openai-compatible-endpoint/v1"
export SKELETON_MODEL="deepseek-v4-pro"
```

主实验示例：

```bash
MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_skeleton_ablation_reuse_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_semantic_skeleton_ablation.sh quick \
  --sample-indices-file /home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_128_20260623_001233/quick_len8192_val1/sample_indices.json \
  --skeleton-file /home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_128_20260623_001233/quick_len8192_val1/skeletons.jsonl \
  --sample-size 128 \
  --val-n 4 \
  --max-new-tokens 16384\
  --max-model-len 20000 \
  --probe-tokens 0 \
  --trajectory-sample-index 0 \
  --logit-size 0 \
  --gpu-memory-utilization 0.9 \
  --seed 0
```

参数要点：

- `--val-n 4`: 每题每个 condition 采样 4 个 rollout，用于 pass@4。
- `--max-new-tokens`: rollout 最大生成长度。
- `--max-model-len`: vLLM rollout 和 HF logit probe 的上下文上限。
- `--probe-tokens 0`: KL/entropy 使用完整回答；正数表示只截断前 N 个 token。
- `--trajectory-sample-index 0`: Phase 3 使用每题第 0 条 `teacher_base` rollout；传 `-1` 使用所有 sample。
- `--logit-size 0`: 不额外 subsample。
- `--gpu-memory-utilization`: 只用于 vLLM rollout 阶段，HF logit probe 不接收这个参数。

## 输入输出

`rollouts.jsonl` 关键字段：

- `problem_id`
- `condition`: `student`, `teacher_base`, `teacher_reference`, `teacher_skeleton`
- `sample_index`
- `problem`
- `solution`
- `ground_truth`
- `full_generation`
- `predicted_answer`
- `correct`
- `completion_tokens`
- `finish_reason`

`logit_probe.jsonl` KL 记录示例：

```json
{
  "record_type": "kl_contrast",
  "logprob_backend": "hf_causal_lm",
  "target_token_source": "target_tail_text",
  "contrast": "teacher_reference_vs_teacher_base",
  "target_condition": "teacher_base",
  "mean_kl": 0.0,
  "kl_per_token": [],
  "top_kl_positions": []
}
```

`logit_probe.jsonl` entropy 记录示例：

```json
{
  "record_type": "rollout_entropy",
  "logprob_backend": "hf_causal_lm",
  "target_token_source": "target_tail_text",
  "condition": "teacher_skeleton",
  "mean_entropy": 0.0,
  "entropy_per_token": []
}
```

## 快速 smoke test

```bash
MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/smoke_skeleton_ablation_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_semantic_skeleton_ablation.sh smoke \
  --sample-size 8 \
  --val-n 1 \
  --max-new-tokens 1024 \
  --max-model-len 20000 \
  --probe-tokens 128 \
  --trajectory-sample-index 0 \
  --logit-size 4 \
  --gpu-memory-utilization 0.9 \
  --seed 0
```

## 注意事项

- Phase 3 已回滚到 HF model probe；它会从文本重新 tokenize，不使用 rollout token ids。
- Phase 3 当前是 HF probe，启动参数以 `python eval/quick_logit_probe.py --help` 为准。
- 如果复用旧 128 题，manifest 里的 `indices` 应该固定不变，不建议每次重新抽样。
