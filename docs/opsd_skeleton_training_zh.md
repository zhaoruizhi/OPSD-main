# OPSD Skeleton-Only 训练说明

## 目标

这套改动给 OPSD 增加一个 `skeleton` teacher context 模式：student 仍然只看题目并进行 on-policy rollout，teacher 不再看完整 reference solution，而是看 semantic skeleton JSON 和 final answer。这样可以直接检验 skeleton 是否能替代完整 reference solution 作为 privileged signal。

原始 OPSD 训练路径保持不变：不传新参数时默认仍是 `teacher_context_mode=reference`。

## 代码说明

### `opsd_train.py`

新增训练参数：

- `--teacher_context_mode {reference,skeleton}`
  - `reference`: 默认值，沿用完整 reference solution。
  - `skeleton`: teacher prompt 使用 semantic skeleton。
- `--skeleton_file PATH`
  - skeleton 模式必传。
  - 文件格式为 `skeletons.jsonl`，用 dataset row index 作为 `problem_id`。
- `--skeleton_subset_policy {error,filter}`
  - `error`: 默认值。只要训练集中有样本缺 skeleton，就直接报错。正式训练必须用这个模式。
  - `filter`: 只保留有 skeleton 的样本。只建议用于 128 题 smoke run。

skeleton 模式下，训练入口会：

1. 读取 `siyanzhao/Openthoughts_math_30k_opsd` 的 train split。
2. 读取 `--skeleton_file`。
3. 按 `problem_id == train row index` 绑定 skeleton。
4. 给每条训练样本增加：
   - `semantic_skeleton`
   - `ground_truth`
5. 再交给 `OPSDTrainer`。

### `opsd_skeleton.py`

这个文件负责训练用 skeleton 数据处理：

- 读取 `skeletons.jsonl`。
- 只保留 `status` 为 `ok` 或 `success` 的记录。
- 兼容 `critical_intermediate`/`critical_intermediates` 和 `check`/`checks` 字段别名。
- 按训练集 row index 绑定 skeleton。
- 在 `error` 模式下检查全量覆盖；在 `filter` 模式下允许子集训练。

### `data_collator.py`

`SelfDistillationDataCollator` 新增 `teacher_context_mode`。

`reference` 模式的 teacher prompt 保持原样：

```text
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{solution}
=== Reference Solution End ===

After reading the reference solution above, ...
Please reason step by step, and put your final answer within \boxed{}.
```

`skeleton` 模式的 teacher prompt 变成：

```text
Problem: {problem}

Final answer: {ground_truth}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{
  "checks": [...],
  "critical_intermediates": [...],
  "key_objects": [...],
  "subgoals": [...],
  "theorem_tags": [...]
}
=== Reference Solution End ===

After reading the reference solution above, ...
Please reason step by step, and put your final answer within \boxed{}.
```

注意：

- `Reference Solution` block 里实际放的是 skeleton JSON。
- skeleton JSON 内会移除 `final_answer` 字段。
- final answer 只单独放在 `Final answer: ...` 行。
- student prompt 完全不变。
- `reason_first=True` 目前只支持 `reference` 模式，不支持 skeleton 模式。

### `opsd_trainer.py`

`OPSDTrainer` 只负责把 `teacher_context_mode` 传给 collator，并保留 `semantic_skeleton`、`ground_truth` 这两个 dataset columns，避免 Trainer 自动裁掉它们。

OPSD loss、student rollout、fixed teacher、LoRA、vLLM 同步逻辑都没有改变。

### `scripts/run_opsd_1b_skeleton.sh`

这是 Qwen3-1.7B skeleton-only 主训练脚本。超参与原始 `scripts/run_opsd_1b.sh` 对齐，只改了：

- `run_config`: `qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005`
- `--teacher_context_mode skeleton`
- `--skeleton_file "$SKELETON_FILE"`
- `--skeleton_subset_policy error`
- `--report_to wandb`

## 训练前准备：生成全量 skeleton

当前 `outputs/opsd_quick/.../skeletons.jsonl` 里的文件通常只有 128 条，用于 ablation 或 smoke run。正式训练需要覆盖完整 train split，约 29.4k 条。

### 1. 创建 skeleton 输出目录

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

FULL_SKEL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/api_full_train_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FULL_SKEL_OUT"
```

### 2. 用 OpenAI-compatible API 生成全量 skeleton

```bash
export SKELETON_API_KEY="你的_API_KEY"
export SKELETON_BASE_URL="https://你的-openai-compatible-endpoint/v1"
export SKELETON_MODEL="deepseek-v4-pro"
export SKELETON_API_CONCURRENCY=8

python eval/generate_semantic_skeletons.py \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --split train \
  --output-file "$FULL_SKEL_OUT/skeletons.jsonl" \
  --skeleton-backend api \
  --skeleton-model "$SKELETON_MODEL" \
  --api-concurrency "$SKELETON_API_CONCURRENCY" \
  --max-tokens 2048
```

不传 `--sample-indices-file` 时，`generate_semantic_skeletons.py` 会遍历完整 train split，直接生成全量 skeleton。`--sample-indices-file` 只用于 ablation、smoke run 或复用固定小样本 manifest。

API backend 会用线程池并发请求，`SKELETON_API_CONCURRENCY` 就是并发数；如果你的 API 有速率限制，可以先把它降到 2 或 4。

### 3. 检查 skeleton 数量

```bash
wc -l "$FULL_SKEL_OUT/skeletons.jsonl"
```

正式训练前，应接近完整 train split 数量。若少于训练集行数，`--skeleton_subset_policy error` 会在启动训练时拦住。

## Smoke run：用 128 条 skeleton 先验链路

如果只是检查代码链路，可以复用已有 128 条 skeleton 文件，并用 `filter` 只训练这些样本。建议同时限制 `--max_steps 1`。

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

SMOKE_SKEL=outputs/opsd_quick/qwen31b_skeleton_ablation_self_20260702_163415/skeletons.jsonl

accelerate launch \
  --config_file accelerate.yaml \
  --num_processes 4 \
  --gradient_accumulation_steps 2 \
  --main_process_port 12949 \
  opsd_train.py \
  --model_name_or_path /data0/shared/Qwen3-1.7B \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 2 \
  --output_dir /data0/siyanz/opsd/ \
  --run_config smoke_qwen31b_skeleton_filter \
  --max_steps 1 \
  --max_completion_length 1024 \
  --save_steps 1 \
  --logging_steps 1 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 20000 \
  --beta 0 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_tensor_parallel_size 1 \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --temperature 1.1 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --fixed_teacher \
  --jsd_token_clip 0.05 \
  --teacher_context_mode skeleton \
  --skeleton_file "$SMOKE_SKEL" \
  --skeleton_subset_policy filter \
  --report_to wandb \
  --wandb_project OPSD
```

## 正式训练

确认 `$FULL_SKEL_OUT/skeletons.jsonl` 是全量 skeleton 后：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

SKELETON_FILE="$FULL_SKEL_OUT/skeletons.jsonl" \
bash scripts/run_opsd_1b_skeleton.sh
```

输出目录默认是：

```text
/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005
```

训练中重点看：

- 终端打印的 `Training rows: 当前数量 / 原始数量`
- wandb run config:
  - `teacher_context_mode=skeleton`
  - `skeleton_file=...`
  - `skeleton_subset_policy=error`
- wandb 曲线:
  - `loss`
  - `on_policy_loss`
  - `grad_norm`
  - `learning_rate`
  - tokens/s 或 step time
- 本地输出:
  - `checkpoint-25`
  - `checkpoint-50`
  - `checkpoint-75`
  - `checkpoint-100`
  - `generations/generations_step_*.json`

## 评估训练效果

评估脚本需要把 `EXP_DIR` 改成 skeleton 训练输出目录：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main/eval
```

可以直接跑单个 checkpoint：

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
  --base_model /data0/shared/Qwen3-1.7B \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 4 \
  --checkpoint_dir /data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100
```

或者按 checkpoint 批量评估：

```bash
BASE_MODEL=/data0/shared/Qwen3-1.7B
EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005

for dataset in aime24 aime25 hmmt25; do
  for step in 25 50 75 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
      --base_model "$BASE_MODEL" \
      --dataset "$dataset" \
      --val_n 12 \
      --temperature 1.0 \
      --tensor_parallel_size 4 \
      --checkpoint_dir "$EXP_DIR/checkpoint-$step" \
      --output_file "eval_results/${dataset}_skeleton_checkpoint_${step}.json"
  done
done
```

核心指标：

- `Average@12`: 所有采样答案的平均正确率。
- `Pass@12`: 每题 12 次采样里至少一次正确的比例。
- `Majority Vote@12`: 多数投票答案正确率。
- `Format rate`: 是否正常输出 `\boxed{}`。

wandb 只能说明训练过程是否稳定；最终效果以 `evaluate_math.py` 的 JSON 和终端指标为准。

## 常见问题

### 启动时报 missing skeletons

说明 `skeletons.jsonl` 没覆盖完整 train split。正式训练应该重新生成全量 skeleton；如果只是 smoke run，改用：

```bash
--skeleton_subset_policy filter
```

### skeleton 文件只有 128 行

这是 ablation/smoke 文件，不适合正式训练。正式训练需要全量 `train` split skeleton。

### 想确认 teacher prompt 没有泄露完整 reference

看 `data_collator.py` 的 skeleton 模式：

- `solution` 不会进入 skeleton teacher prompt。
- `semantic_skeleton` 会被 JSON dump 后放入 `Reference Solution` block。
- `final_answer` 从 JSON 中移除，只作为 `Final answer: ...` 单独出现。

### 想和原始 OPSD 公平对比

保持下面配置一致：

- base model: `/data0/shared/Qwen3-1.7B`
- LoRA: `r=64, alpha=128`
- rollout: `max_completion_length=1024`
- sampling: `temperature=1.1, top_p=0.95, top_k=20`
- loss: `beta=0, jsd_token_clip=0.05`
- teacher: `fixed_teacher`
- checkpoints: `25/50/75/100`
- eval: `temperature=1.0, val_n=12`

唯一改变应是：

```bash
--teacher_context_mode skeleton
```
