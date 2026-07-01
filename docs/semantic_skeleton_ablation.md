# Semantic Skeleton Ablation 实验说明

## 实验目标

这个实验比较完整 reference solution 和抽象 semantic skeleton 对 thinking teacher 的影响。主实验包含四路：

- `student`: non-thinking，prompt 只包含 problem。
- `teacher_base`: thinking，prompt 只包含 problem。
- `teacher_reference`: thinking，prompt 包含完整 reference solution 和 final answer。
- `teacher_skeleton`: thinking，prompt 使用 reference solution 外层格式，内容为 style-neutral semantic skeleton JSON 和 final answer。

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
   - rollout 记录保存 `prompt_token_ids` 和 `completion_token_ids`。

4. Phase 3: `eval/quick_logit_probe.py`
   - 使用 HuggingFace `AutoModelForCausalLM` dense forward 计算 full-response KL 和 entropy。
   - target 轨迹优先来自 `completion_token_ids`；旧产物缺少 token ids 时才回退到 `full_generation` / `target_tail_text` 文本重新 tokenize。
   - 输出 `$OUT/logit_probe.jsonl`、`$OUT/logit_summary.json`。

## Prompt 是怎么使用 skeleton 的

`teacher_reference` 使用完整 reference solution prompt。`teacher_skeleton` 使用相同 reference solution 外层格式，`Reference Solution` 块中放入 skeleton JSON：

```text
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{skeleton_json_without_final_answer}
=== Reference Solution End ===

Final answer: {answer}

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and don't be afraid to backtrack or reconsider if something doesn't work out:

Please reason step by step, and put your final answer within \boxed{}.
```

注意：

- `teacher_skeleton` prompt 不包含完整 reference solution；`Reference Solution` 块内是 skeleton JSON。
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
  --gpus "4 5 6 7" \
  --sample-indices-file /home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_128_20260623_001233/quick_len8192_val1/sample_indices.json \
  --skeleton-file /home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_128_20260623_001233/quick_len8192_val1/skeletons.jsonl \
  --sample-size 128 \
  --val-n 4 \
  --max-new-tokens 16384 \
  --max-model-len 20000 \
  --probe-tokens 0 \
  --trajectory-sample-index 0 \
  --logit-size 0 \
  --gpu-memory-utilization 0.75 \
  --hf-device-map cuda \
  --seed 0
```

参数要点：

- `--gpus "4 5 6 7"`: 指定本次实验使用哪几张物理 GPU；默认也是 `4 5 6 7`。例如只用两张卡可以传 `--gpus "5 7"`。
- `--val-n 4`: 每题每个 condition 采样 4 个 rollout，用于 pass@4。
- `--max-new-tokens`: rollout 最大生成长度。
- `--max-model-len`: vLLM rollout 和 HF logit probe 的上下文上限。
- `--probe-tokens 0`: KL/entropy 使用完整回答；正数表示只截断前 N 个 token。
- `--trajectory-sample-index 0`: Phase 3 使用每题第 0 条 `teacher_base` rollout；传 `-1` 使用所有 sample。
- `--logit-size 0`: 不额外 subsample。
- `--gpu-memory-utilization`: 只用于 vLLM rollout 阶段，HF logit probe 不接收这个参数。
- `--hf-device-map cuda`: Phase 3 的 HF probe 固定加载到当前进程可见的 `cuda:0`。因为每个子进程都设置了 `CUDA_VISIBLE_DEVICES=$gpu`，所以这里的 `cuda:0` 实际对应 `--gpus` 中分配给该 shard 的物理 GPU。

## GPU 与 shard 映射

脚本会把 `--gpus` 拆成 GPU 列表，并自动设置 `--num-shards` 为 GPU 数量。shard id 使用列表下标，而不是物理 GPU id：

- `--gpus "4 5 6 7"` 会启动 4 个进程，shard id 分别是 `0,1,2,3`，物理 GPU 分别是 `4,5,6,7`。
- `--gpus "5 7"` 会启动 2 个进程，shard id 分别是 `0,1`，物理 GPU 分别是 `5,7`。
- 分片文件名包含物理 GPU id，例如 `rollout_shard4.jsonl`、`logit_probe_shard4.jsonl`，方便回看是哪张卡跑出来的。

多人共用服务器时，建议把 `--gpu-memory-utilization` 从默认 `0.9` 降到 `0.75` 或 `0.8`。如果 vLLM 报 `EngineCore_DP0 died unexpectedly`，并且日志里出现 free memory 小于目标 utilization 的提示，通常是 Phase 2 rollout 启动时某张 GPU 可用显存不够，不是 Phase 3 KL 计算逻辑本身坏了。

当前默认会保留完整产物，不会跳过 rollout entropy。正常跑完后至少应有：

- `$OUT/rollouts.jsonl`
- `$OUT/rollout_summary.json`
- `$OUT/logit_probe.jsonl`
- `$OUT/logit_summary.json`

## 输入输出

`rollouts.jsonl` 关键字段：

- `problem_id`
- `condition`: `student`, `teacher_base`, `teacher_reference`, `teacher_skeleton`
- `sample_index`
- `problem`
- `solution`
- `ground_truth`
- `full_generation`
- `prompt_token_ids`
- `completion_token_ids`
- `predicted_answer`
- `correct`
- `completion_tokens`
- `finish_reason`

`logit_probe.jsonl` KL 记录示例：

```json
{
  "record_type": "kl_contrast",
  "logprob_backend": "hf_causal_lm",
  "target_token_source": "completion_token_ids",
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
  "target_token_source": "completion_token_ids",
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
  --gpus "4" \
  --sample-size 8 \
  --val-n 1 \
  --max-new-tokens 1024 \
  --max-model-len 20000 \
  --probe-tokens 128 \
  --trajectory-sample-index 0 \
  --logit-size 4 \
  --gpu-memory-utilization 0.75 \
  --hf-device-map cuda \
  --seed 0
```

## 注意事项

- Phase 3 使用 HF model probe；新 rollout 产物会直接使用 `prompt_token_ids` 和 `completion_token_ids`，旧产物缺少 token ids 时才从文本重新 tokenize。
- Phase 3 当前是 HF probe，启动参数以 `python eval/quick_logit_probe.py --help` 为准。
- 如果复用旧 128 题，manifest 里的 `indices` 应该固定不变，不建议每次重新抽样。
- 如果只想临时加速检查 Phase 3，可以显式传 `--skip-rollout-entropy`，但正式对齐旧产物时不要传这个参数。
