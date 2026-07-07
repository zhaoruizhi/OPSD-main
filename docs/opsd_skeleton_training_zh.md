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
  - `filter`: 只保留有 skeleton 的样本。只建议用于临时小样本调试，不用于正式对比。

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

正式训练需要覆盖完整 train split，约 29.4k 条。`outputs/opsd_quick/.../skeletons.jsonl` 这类小样本文件不适合正式训练。

### 1. 创建 skeleton 输出目录

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

FULL_SKEL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/api_full_train_$(date +%Y%m%d_%H%M%S)
mkdir -p "$FULL_SKEL_OUT"
```

### 2. 用 OpenAI-compatible API 生成全量 skeleton

```bash
export SKELETON_API_KEY="你的_API_KEY"
export SKELETON_BASE_URL="https://api.deepseek.com"
export SKELETON_MODEL="deepseek-v4-pro"
export SKELETON_API_CONCURRENCY=2
export SKELETON_TIMEOUT=300
export SKELETON_FLUSH_EVERY=10
export SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES=50
export SKELETON_RESPONSE_FORMAT_JSON=1
export SKELETON_API_DISABLE_THINKING=1

python eval/generate_semantic_skeletons.py \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --split train \
  --output-file "$FULL_SKEL_OUT/skeletons.jsonl" \
  --skeleton-backend api \
  --skeleton-model "$SKELETON_MODEL" \
  --api-concurrency "$SKELETON_API_CONCURRENCY" \
  --timeout "$SKELETON_TIMEOUT" \
  --flush-every "$SKELETON_FLUSH_EVERY" \
  --abort-after-consecutive-failures "$SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES" \
  --response-format-json \
  --api-disable-thinking \
  --max-tokens 2048
```

不传 `--sample-indices-file` 时，`generate_semantic_skeletons.py` 会遍历完整 train split，直接生成全量 skeleton。`--sample-indices-file` 只用于 ablation 或复用固定小样本 manifest。

API backend 会用线程池并发请求，`SKELETON_API_CONCURRENCY` 就是并发数。全量生成建议先用 `2` 跑通接口稳定性，再逐步升到 `4` 或 `8`；不要一开始开到几十或上百。`SKELETON_RESPONSE_FORMAT_JSON=1` 会请求 OpenAI-compatible JSON object mode，通常能减少空响应和非 JSON 输出；如果你的 endpoint 不支持这个参数并返回 `HTTP Error 400/422`，把它改成 `0` 或删掉 `--response-format-json` 后重跑。DeepSeek 官方 API 默认开启 thinking mode，生成 skeleton 时应使用 `SKELETON_API_DISABLE_THINKING=1` 或 `--api-disable-thinking`，否则输出 token 可能全部消耗在 `reasoning_content`，导致 `content` 为空。

如果长跑过程中在 `ssl.py` / `http.client.py` 附近报 read timeout、connection reset 或 HTTPS 连接中断，通常是 API endpoint 在高并发下限流、排队或断开连接，不是 dataset 行本身坏了。建议先把并发从 100 降到 2、4 或 8，并提高重试次数：

```bash
export SKELETON_API_CONCURRENCY=16
export SKELETON_FLUSH_EVERY=10
export SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES=50
export SKELETON_API_DISABLE_THINKING=1

python eval/generate_semantic_skeletons.py \
  --dataset siyanzhao/Openthoughts_math_30k_opsd \
  --split train \
  --output-file "$FULL_SKEL_OUT/skeletons.jsonl" \
  --skeleton-backend api \
  --skeleton-model "$SKELETON_MODEL" \
  --api-concurrency "$SKELETON_API_CONCURRENCY" \
  --flush-every "$SKELETON_FLUSH_EVERY" \
  --abort-after-consecutive-failures "$SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES" \
  --api-disable-thinking \
  --timeout 300 \
  --max-retries 5 \
  --max-tokens 2048
```

脚本默认会 resume，并会先清理已有 `--output-file`：只保留每个 `problem_id` 的第一条 `status=ok` 记录，删除旧的 `status=error` 和重复记录。继续生成时，脚本会先从旧文件最大 `problem_id` 之后往后跑；旧文件里只有 error、没有 ok 的样本会排到后面修复。默认情况下，新运行不会再向主 `skeletons.jsonl` 写 `status=error`；失败尝试会写入 sidecar 文件 `skeletons.failures.jsonl`，失败样本会让出 worker 并推迟到下一轮 retry pass，避免少数坏样本堵住全局并发。只有显式传 `--allow-error-records` 才会恢复旧行为。

如果 sidecar 里出现 `error: API returned empty assistant content` 且 `api_finish_reason: length`，同时 `raw_response` 里有很长的 `reasoning_content`，说明 DeepSeek thinking mode 把 `--max-tokens` 用完了，还没输出最终 JSON。解决办法是加 `--api-disable-thinking`。如果 `content` 仍为空但没有长 `reasoning_content`，再检查 endpoint/model 与 chat schema 或 JSON 输出设置是否匹配。新版脚本会额外记录 `api_finish_reason`、`api_message_keys`、`api_choice_keys`、`api_body_keys`，并把完整 API body 截断保存到 `raw_response`。如果连续失败达到 `--abort-after-consecutive-failures`，脚本会停止，让你先检查这些诊断字段；确认只是短暂服务波动时，可以传 `--abort-after-consecutive-failures 0` 关闭保护。

### 3. 检查 skeleton 数量

```bash
wc -l "$FULL_SKEL_OUT/skeletons.jsonl"
```

正式训练前，应接近完整 train split 数量。若少于训练集行数，`--skeleton_subset_policy error` 会在启动训练时拦住。

## 全量训练对比：reference baseline vs skeleton

这部分要跑两次完整训练：

1. 先跑原始 OPSD reference baseline，复现完整 reference solution teacher。
2. 再跑 skeleton teacher，除 teacher context 和 skeleton 文件外，其他训练超参保持一致。

两次训练都建议使用同一组 GPU、同一个 wandb project，并用不同的 `run_config` 区分。

### 1. 登录 wandb

首次在服务器上使用 wandb 需要登录。已经登录过可以用 `wandb status` 检查。

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

wandb login
wandb status
```

如果你的账号在团队/组织 entity 下，可以额外设置：

```bash
export WANDB_ENTITY=你的_wandb_entity
```

脚本里已经传了 `--report_to wandb --wandb_project OPSD`。训练启动后，在 wandb 的 `OPSD` project 里应看到两条 run：

- `qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005`
- `qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005`

### 2. 指定训练 GPU

两个训练脚本都支持用 `TRAIN_GPU_IDS` 指定可见 GPU，用 `NUM_PROCESSES` 指定 accelerate 进程数。比如使用物理 GPU 0、1、2、3：

```bash
export TRAIN_GPU_IDS=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=12949
```

如果换成别的卡，比如 4、5、6、7：

```bash
export TRAIN_GPU_IDS=4,5,6,7
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=12949
```

### 3. 跑 reference baseline 全量训练

这一步复现原始 OPSD：teacher 继续使用完整 reference solution。不要传 `SKELETON_FILE`。

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

export TRAIN_GPU_IDS=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=12949

bash scripts/run_opsd_1b.sh
```

reference baseline 输出目录默认是：

```text
/data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005
```

### 4. 跑 skeleton 全量训练

确认 `$FULL_SKEL_OUT/skeletons.jsonl` 是全量 skeleton 后，再启动 skeleton 训练。这里必须使用 `skeleton_subset_policy=error`，如果 skeleton 没覆盖全量 train split，训练应直接报错。

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

export TRAIN_GPU_IDS=0,1,2,3
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=12949
export SKELETON_FILE="$FULL_SKEL_OUT/skeletons.jsonl"

bash scripts/run_opsd_1b_skeleton.sh
```

skeleton 输出目录默认是：

```text
/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005
```

### 5. 对比 wandb 曲线

重点把这两条 run 放到同一个 wandb panel 里比较：

- `loss`
- `on_policy_loss`
- `grad_norm`
- `learning_rate`
- tokens/s、samples/s、steps/s 或 step time

对比时重点确认：

- 两条 run 的 learning rate schedule 是否一致。
- `grad_norm` 是否出现明显爆炸或长期异常。
- `on_policy_loss` 是否在同一量级。
- skeleton run 的 step time 是否明显变慢；如果变慢，通常是 prompt/token 长度或 vLLM 同步开销变化。
- 本地输出是否都有 `checkpoint-25/50/75/100` 和 `generations/generations_step_*.json`。

## 评估训练效果

训练曲线只能说明优化过程是否稳定；最终效果仍以 `evaluate_math.py` 的结果为准。建议 reference baseline 和 skeleton 使用同一批 checkpoint、同一批数据集、同一组 GPU、同一组采样参数评估。

评估脚本需要把 `EXP_DIR` 分别改成 reference 或 skeleton 训练输出目录：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main/eval
```

可以先评估单个 checkpoint：

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
  --base_model /data0/shared/Qwen3-1.7B \
  --dataset aime24 \
  --val_n 12 \
  --temperature 1.0 \
  --tensor_parallel_size 4 \
  --checkpoint_dir /data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100
```

或者按 checkpoint 批量评估两组训练：

```bash
BASE_MODEL=/data0/shared/Qwen3-1.7B
REFERENCE_EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005
SKELETON_EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005

for condition in reference skeleton; do
  if [[ "$condition" == "reference" ]]; then
    EXP_DIR="$REFERENCE_EXP_DIR"
  else
    EXP_DIR="$SKELETON_EXP_DIR"
  fi

  for dataset in aime24 aime25 hmmt25; do
    for step in 25 50 75 100; do
      NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "$dataset" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 4 \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step" \
        --output_file "eval_results/${dataset}_${condition}_checkpoint_${step}.json"
    done
  done
done
```

如果只想先评估 skeleton，可以保留原来的单组循环：

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

说明 `skeletons.jsonl` 没覆盖完整 train split。正式训练必须重新生成全量 skeleton，不要用 `filter` 绕过。

### skeleton 文件只有 128 行

这是小样本文件，不适合正式训练。正式训练需要全量 `train` split skeleton。

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
