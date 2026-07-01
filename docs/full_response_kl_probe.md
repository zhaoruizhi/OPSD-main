# HF Full-Response KL Probe

## 背景

当前 `eval/quick_logit_probe.py` 使用 HuggingFace causal LM probe。Phase 3 优先使用 rollout 中保存的 prompt token ids 和 completion token ids，然后做 HF dense forward 计算 KL/entropy；旧产物缺少 token ids 时才回退到文本重建。

这意味着：

- rollout 仍由 vLLM 生成。
- logit probe 使用 HF `AutoModelForCausalLM` 对 prompt token ids + completion token ids 做 dense forward。
- target 轨迹优先来自 `completion_token_ids`；旧 `rollouts.jsonl` 缺少该字段时，才来自 `full_generation` / `target_tail_text` 文本重新 tokenize。
- 输出 backend 为 `hf_causal_lm`。

## 数据流

1. Phase 2 rollout: `eval/quick_rollout_openthoughts.py`
   - 用 vLLM 生成 `student`、`teacher_base`、`teacher_reference`、`teacher_skeleton`。
   - 输出 `rollouts.jsonl`、`rollout_summary.json`。

2. Phase 3 HF logit probe: `eval/quick_logit_probe.py`
   - 读取 `rollouts.jsonl`。
   - 默认沿 `teacher_base` 的 `completion_token_ids` 做 target trajectory。
   - 分别读取 `teacher_base`、`teacher_reference`、`teacher_skeleton` 的 `prompt_token_ids`；旧产物缺失时才重建 prompt。
   - 用 HF dense logits 计算：
     - `teacher_reference_vs_teacher_base`
     - `teacher_skeleton_vs_teacher_base`
   - 四路 rollout entropy 也用 HF dense logits 计算。

## 终端启动方式

推荐从仓库根目录运行：

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

`--gpus` 接收物理 GPU id 列表。脚本会按列表长度自动设置 shard 数，并在 Phase 2 rollout 和 Phase 3 HF probe 中分别设置 `CUDA_VISIBLE_DEVICES=$gpu`。因此 `--hf-device-map cuda` 下，每个 HF probe 子进程看到的 `cuda:0` 就是分配给它的那张物理 GPU。

单独重跑 logit probe：

```bash
CUDA_VISIBLE_DEVICES=4 python eval/quick_logit_probe.py \
  --model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --rollout-file /path/to/rollouts.jsonl \
  --skeleton-file /path/to/skeletons.jsonl \
  --trajectory-condition teacher_base \
  --trajectory-sample-index 0 \
  --probe-tokens 0 \
  --logit-size 0 \
  --top-k 20 \
  --max-context-tokens 20000 \
  --hf-device-map cuda \
  --output-file /path/to/logit_probe.jsonl \
  --summary-file /path/to/logit_summary.json
```

## 注意事项

- 当前 probe 速度应接近旧 HF 版本，但分布来自 HF forward，不是 vLLM sampler 内部返回的 logprobs。
- 正式对齐旧产物时不要传 `--skip-rollout-entropy`；当前主脚本默认会生成 rollout entropy、`logit_probe.jsonl` 和 `logit_summary.json`。
- 多人共用服务器时，Phase 2 vLLM rollout 可以把 `--gpu-memory-utilization` 降到 `0.75` 或 `0.8`，避免某张卡因启动显存阈值不满足导致 `EngineCore_DP0 died unexpectedly`。
- 如果后续要重新做 vLLM-exact probe，建议单独开 git 分支并先用小样本验证显存和速度。
