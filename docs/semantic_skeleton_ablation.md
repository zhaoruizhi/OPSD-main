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
   - 默认 `--skeleton-backend api`，走 OpenAI-compatible API。
   - 新增 `--skeleton-backend vllm`，走本地 Qwen/vLLM 生成 skeleton。此模式下 `--skeleton-model` 通常设为同一个 Qwen3-1.7B 模型路径。

3. Phase 2: `eval/quick_rollout_openthoughts.py`
   - 用 vLLM 生成四路 rollout。
   - 输出 `$OUT/rollout_shard*.jsonl`、`$OUT/rollouts.jsonl`、`$OUT/rollout_summary.json`。
   - rollout 记录保存 `prompt_token_ids` 和 `completion_token_ids`。

4. Phase 3: `eval/quick_logit_probe.py`
   - 使用 HuggingFace `AutoModelForCausalLM` dense forward 计算 full-response KL 和 entropy。
   - target 轨迹优先来自 `completion_token_ids`；旧产物缺少 token ids 时才回退到 `full_generation` / `target_tail_text` 文本重新 tokenize。
   - 输出 `$OUT/logit_probe.jsonl`、`$OUT/logit_summary.json`。

## Prompt 是怎么使用 skeleton 的

`teacher_reference` 和 `teacher_skeleton` 使用同一个 privileged prompt 模板，只替换 `{solution}`：

- `teacher_reference`: `{solution}` 是完整 reference solution。
- `teacher_skeleton`: `{solution}` 是去掉 `final_answer` 字段后的 semantic skeleton JSON。

```text
Problem: {problem}

Final answer: {ground_truth}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the reasoning behind each step - do not copy or paraphrase it. Now, using your own words and independent reasoning, derive the same final answer to the problem above. Think step by step, explore different approaches, and don't be afraid to backtrack or reconsider if something doesn't work out:

Please reason step by step, and put your final answer within \boxed{}.
```

注意：

- `teacher_skeleton` prompt 不包含完整 reference solution；`Reference Solution` 块内是 skeleton JSON。
- skeleton JSON 不包含 `final_answer` 字段；ground truth 单独放在 `Final answer: ...` 行。
- `teacher_base` 只包含 problem，不包含 privileged info。

## 本次增量改动：skeleton 生成 backend 可选

原来的 skeleton 生成只支持 API backend，通常用 DeepSeek/OpenAI-compatible endpoint 生成 `$OUT/skeletons.jsonl`。这可能把外部大模型的风格、抽象偏好或解析错误引入实验。

现在 Phase 1 增加了一个可选 backend：

- `--skeleton-backend api`: 默认值，行为与旧实验保持一致，需要 `SKELETON_API_KEY` 和 `SKELETON_BASE_URL`。
- `--skeleton-backend vllm`: 使用本地 vLLM 加载 `--skeleton-model`，让 Qwen3-1.7B 自己把 reference solution 编译成 semantic skeleton。

新增参数：

- `--skeleton-backend {api,vllm}`: 选择 skeleton 生成方式。
- `--skeleton-model PATH_OR_API_MODEL`: API 模式下是 API model name；vLLM 模式下是本地模型路径。vLLM 模式如果不传，脚本默认使用 `--model`。
- `--skeleton-gpus "4"`: vLLM skeleton 生成阶段可见 GPU。默认使用 `--gpus` 的第一张卡。
- `--skeleton-vllm-tensor-parallel-size 1`: skeleton vLLM 的 tensor parallel size。
- `--skeleton-vllm-gpu-memory-utilization 0.75`: skeleton vLLM 显存占用比例，默认继承 `--gpu-memory-utilization`。
- `--skeleton-vllm-max-model-len 20000`: skeleton vLLM 上下文长度，默认继承 `--max-model-len`。
- `--skeleton-vllm-top-p 1.0`、`--skeleton-vllm-top-k -1`: skeleton 生成采样参数。
- `--skeleton-enable-thinking`: skeleton 生成时开启 Qwen thinking。默认关闭，目的是让 JSON-only 输出更稳定。

输出记录会额外保存 `skeleton_backend` 字段，方便区分 skeleton 是 API 生成还是 vLLM 自生成。

## 正式实验怎么跑

### 推荐：Qwen/vLLM 自生成 skeleton 完整命令

这条命令会完整跑 Phase 0-3：抽样 manifest、本地 Qwen/vLLM 生成 skeleton、四路 rollout、full-response KL/entropy probe。

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

MODEL=/data0/shared/Qwen3-1.7B \
OUT=/data1/opsd_quick/qwen31b_skeleton_ablation_qwen_skeleton_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_semantic_skeleton_ablation.sh quick \
  --model /data0/shared/Qwen3-1.7B \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --split train \
  --gpus "4 5 6 7" \
  --skeleton-backend vllm \
  --skeleton-model /data0/shared/Qwen3-1.7B \
  --skeleton-gpus "4" \
  --skeleton-vllm-tensor-parallel-size 1 \
  --skeleton-vllm-gpu-memory-utilization 0.75 \
  --skeleton-vllm-max-model-len 20000 \
  --skeleton-vllm-top-p 1.0 \
  --skeleton-vllm-top-k -1 \
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

如果要复用固定 128 题 sample manifest，但重新用 Qwen 自生成 skeleton，把 `--sample-indices-file` 加进去即可：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

MODEL=/data0/shared/Qwen3-1.7B \
OUT=/data1/opsd_quick/qwen31b_skeleton_ablation_qwen_skeleton_reuse_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_semantic_skeleton_ablation.sh quick \
  --model /data0/shared/Qwen3-1.7B \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --split train \
  --gpus "4 5 6 7" \
  --sample-indices-file /path/to/sample_indices.json \
  --skeleton-backend vllm \
  --skeleton-model /data0/shared/Qwen3-1.7B \
  --skeleton-gpus "4" \
  --skeleton-vllm-tensor-parallel-size 1 \
  --skeleton-vllm-gpu-memory-utilization 0.75 \
  --skeleton-vllm-max-model-len 20000 \
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

注意这里不要传 `--skeleton-file`，因为本实验目标正是重新生成 skeleton。只有在复用旧 skeleton 时才传 `--skeleton-file`。

### 旧路径：API 生成或复用 skeleton

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
- `--skeleton-backend vllm`: 使用本地 Qwen/vLLM 自生成 skeleton；不传则默认走 API backend。
- `--skeleton-gpus "4"`: 只影响 Phase 1 skeleton vLLM 生成。Phase 2/3 仍由 `--gpus` 控制。
- `--skeleton-model /data0/shared/Qwen3-1.7B`: vLLM skeleton compiler 使用的模型路径。
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

如果 Phase 1 报：

```text
RuntimeError: semantic skeleton generation failed for N examples
```

并且 `$OUT/skeletons.jsonl` 里失败记录的 `error` 是 `Invalid \escape`，通常说明 vLLM 本地模型已经成功加载并完成生成，但输出的 JSON 字符串里包含未转义的 LaTeX 反斜杠，例如 `\left`、`\frac`、`\pmod`、`\geq` 或 `\$`。这类输出人眼看接近 JSON，但严格 `json.loads` 会拒绝。当前代码已对 skeleton 解析增加容错：会剥离可选的 ```json code fence，并修复 JSON 字符串中的 LaTeX 风格单反斜杠；prompt 里也额外要求本地模型尽量使用 plain text 或对反斜杠做 JSON 转义。遇到旧版本生成的失败文件时，建议更新代码后重新跑 Phase 1，或者把已有 `raw_response` 重新解析后生成一份修复后的 `skeletons.jsonl`。

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

Qwen/vLLM 自生成 skeleton 的 smoke test：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

MODEL=/data0/shared/Qwen3-1.7B \
OUT=/data1/opsd_quick/smoke_skeleton_ablation_qwen_skeleton_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_semantic_skeleton_ablation.sh smoke \
  --model /data0/shared/Qwen3-1.7B \
  --gpus "4" \
  --skeleton-backend vllm \
  --skeleton-model /data0/shared/Qwen3-1.7B \
  --skeleton-gpus "4" \
  --skeleton-vllm-tensor-parallel-size 1 \
  --skeleton-vllm-gpu-memory-utilization 0.75 \
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

旧 API/复用 skeleton 路径的 smoke test：

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
- 如果比较 DeepSeek skeleton 与 Qwen 自生成 skeleton，除了 `--skeleton-backend/--skeleton-model` 之外，尽量保持 sample manifest、rollout 参数、KL probe 参数完全一致。
- 如果只想临时加速检查 Phase 3，可以显式传 `--skip-rollout-entropy`，但正式对齐旧产物时不要传这个参数。
