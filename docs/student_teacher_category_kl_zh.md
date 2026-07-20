# Student vs Reference/Skeleton Teacher 分类 KL 说明

## 目标

这个实验用于在 10 个 problem 上测：

- target trajectory：`student` 自己生成的 completion tokens。
- baseline distribution：student prompt 作为条件上下文，即只看 problem 的 `P_student`。
- teacher distributions：
  - `teacher_reference`：看 problem、final answer、完整 reference solution。
  - `teacher_skeleton`：看 problem、final answer、semantic skeleton JSON。
- 指标：沿 student 轨迹逐 token 计算 `KL(P_teacher || P_student)`，再按 target token 文本分成 `style`、`math`、`other` 三类，输出每类平均 KL。

这里不是计算 prompt token 的 KL。prompt 只决定条件上下文；真正被打分的 target tokens 是 student rollout 的 completion tokens。teacher 沿 OPSD 口径指“同一套模型权重在 privileged teacher prompt 下的分布”。如果 checkpoint 是 LoRA，脚本会用 `--base-model` 加载 base Qwen3，再用 `--checkpoint-dir` 加载 adapter。

## 代码改动

### `eval/quick_opsd_common.py`

新增 token 分类逻辑：

- `token_category(token_text) -> style|math|other`
- `summarize_token_category_values(token_texts, values)`
- `aggregate_token_category_kl(records)`

Style/Math 词表来自截图。分类时会对 decoded token 做：

1. 去掉首尾空白和常见 tokenizer word-boundary marker。
2. 小写化。
3. 去掉词首/词尾常见标点。
4. 先判定 style，再判定 math，剩下归为 other。

分类完全按照论文图片中的 Style/Math 词表执行；不额外把数字、数学符号、`\frac`、`\sqrt` 等启发式归入 math。

### `eval/quick_logit_probe.py`

新增参数：

- `--base-model`
- `--checkpoint-dir`
- `--baseline-condition`
- `--teacher-condition`，可重复传。
- `--student-enable-thinking`
- `--require-context-rollouts`

本实验使用：

```bash
--trajectory-condition student \
--baseline-condition student \
--teacher-condition teacher_reference \
--teacher-condition teacher_skeleton
```

这样会生成两个 contrast：

- `teacher_reference_vs_student`
- `teacher_skeleton_vs_student`

每条 `kl_contrast` 记录新增：

```json
"token_category_kl": {
  "style": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0},
  "math": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0},
  "other": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0}
}
```

`logit_summary.json` / `student_teacher_category_kl_summary.json` 会跨所有 case 做 token 加权平均，并额外给出便捷字段：

- `mean_style_kl`
- `mean_math_kl`
- `mean_other_kl`
- `mean_style_kl_share`
- `mean_math_kl_share`
- `mean_other_kl_share`
- `style_token_count`
- `math_token_count`
- `other_token_count`

### `eval/quick_rollout_openthoughts.py`

新增参数：

- `--base-model`
- `--checkpoint-dir`
- `--student-enable-thinking`

默认 student 是 TM-off，即 `enable_thinking=False`。传 `--student-enable-thinking` 后 student rollout 使用 thinking chat template，输出记录中的 `enable_thinking` 也会写成 `true`，供 KL prompt fallback 使用。

### `scripts/run_student_teacher_category_kl.sh`

新增一键入口，默认只跑 10 个 problem：

1. 生成或复用 sample manifest。
2. 只生成 `student` rollout。
3. 沿 student tokens 计算 reference/skeleton teacher 相对 student 的分类 KL。

默认：

- `SAMPLE_SIZE=10`
- `VAL_N=1`
- `--student-tm off` 时 `MAX_NEW_TOKENS=1024`
- `--student-tm on` 时 `MAX_NEW_TOKENS=16384`

## 运行命令

### TM-off student，Qwen3-1.7B，10 题

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main

BASE_MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
CHECKPOINT_DIR=/home/ruizzhao/OPSD-main/outputs/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-100 \
SKELETON_FILE=/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_tmoff_student_teacher_category_kl_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_student_teacher_category_kl.sh \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --seed 0
```

如果你的 checkpoint 已经是合并后的完整模型，不需要 `CHECKPOINT_DIR`：

```bash
BASE_MODEL=/path/to/merged-qwen31b-checkpoint \
SKELETON_FILE=/path/to/skeletons.jsonl \
bash scripts/run_student_teacher_category_kl.sh --student-tm off --sample-size 10 --gpu-ids "4"
```

### 后续切换 TM-on student

```bash
BASE_MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
CHECKPOINT_DIR=/path/to/tm_on_or_target_checkpoint/checkpoint-100 \
SKELETON_FILE=/path/to/skeletons.jsonl \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_tmon_student_teacher_category_kl_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_student_teacher_category_kl.sh \
  --student-tm on \
  --sample-size 10 \
  --gpu-ids "4" \
  --max-model-len 32768 \
  --hf-device-map cuda \
  --seed 0
```

`--student-tm on` 会自动把 `max-new-tokens` 设为 `16384`。如果你想显式覆盖，可以加：

```bash
--max-new-tokens 16384
```

### 复用固定 10 题 sample

```bash
bash scripts/run_student_teacher_category_kl.sh \
  --student-tm off \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --skeleton-file /path/to/skeletons.jsonl \
  --sample-indices-file /path/to/sample_indices.json \
  --sample-size 10 \
  --gpu-ids "4"
```

### 只重跑 KL，不重新生成 student rollout

```bash
CUDA_VISIBLE_DEVICES=4 python eval/quick_logit_probe.py \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --rollout-file /path/to/student_rollouts.jsonl \
  --skeleton-file /path/to/skeletons.jsonl \
  --trajectory-condition student \
  --baseline-condition student \
  --teacher-condition teacher_reference \
  --teacher-condition teacher_skeleton \
  --trajectory-sample-index 0 \
  --logit-size 0 \
  --probe-tokens 0 \
  --top-k 20 \
  --max-context-tokens 20000 \
  --skip-rollout-entropy \
  --hf-device-map cuda \
  --output-file /path/to/student_teacher_category_kl.jsonl \
  --summary-file /path/to/student_teacher_category_kl_summary.json
```

如果这批 student rollout 是 TM-on 但缺少 `prompt_token_ids`，补上：

```bash
--student-enable-thinking
```

新生成的 rollout 会保存 `prompt_token_ids`，一般不需要靠 fallback 重建 prompt。

## 输出怎么看

脚本结束后重点看：

```text
$OUT/student_teacher_category_kl_summary.json
```

核心字段：

```json
{
  "contrasts": {
    "teacher_reference_vs_student": {
      "num_cases": 10,
      "mean_style_kl": 0.0,
      "mean_math_kl": 0.0,
      "mean_other_kl": 0.0,
      "mean_style_kl_share": 0.0,
      "mean_math_kl_share": 0.0,
      "mean_other_kl_share": 0.0,
      "token_category_kl": {
        "style": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0},
        "math": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0},
        "other": {"num_tokens": 0, "sum_kl": 0.0, "mean_kl": 0.0}
      }
    },
    "teacher_skeleton_vs_student": {
      "num_cases": 10,
      "mean_style_kl": 0.0,
      "mean_math_kl": 0.0,
      "mean_other_kl": 0.0,
      "mean_style_kl_share": 0.0,
      "mean_math_kl_share": 0.0,
      "mean_other_kl_share": 0.0
    }
  }
}
```

如果要做表格，取每个 contrast 的：

- Style: `mean_style_kl`
- Math: `mean_math_kl`
- Other: `mean_other_kl`

如果要看 KL mass 占比，取 `mean_style_kl_share`、`mean_math_kl_share`、`mean_other_kl_share`。

## 注意事项

- `--skeleton-file` 必传。脚本默认复用已有 skeleton，不自动调用 API 重新生成。
- `problem_id` 必须和 dataset train row index 对齐；全量 skeleton 文件通常满足这个条件。
- `--probe-tokens 0` 表示 KL 覆盖完整 student response；如果只想快速 smoke test，可以设成 `128`。
- `--gpu-ids "4 5"` 会把 10 题按 shard 分到两张卡，输出 shard JSONL 后自动合并；旧参数名 `--gpus "4 5"` 仍兼容。
- `--hf-device-map cuda` 下，每个 KL 子进程只看到 `CUDA_VISIBLE_DEVICES=$gpu`，因此代码里的 `cuda:0` 对应分配到的物理 GPU。

## Phase 3：在全局 Top-KL 位置续写 teacher

完整脚本现在默认在 KL 计算完成后继续运行 `scripts/run_teacher_spike_continuations.sh`：

- 从所有 KL shards 原子重建并校验 aggregate；
- 选择全局 Top 10 唯一 token 位置；
- 在 student token 之前分叉；
- reference teacher 和 skeleton teacher 各 greedy 续写 20 tokens；
- 生成 student/reference/skeleton 三列 HTML 报告。

完整流程可增加：

```bash
--teacher-continuation-top-n 10 \
--teacher-continuation-max-new-tokens 20
```

如果暂时只需要旧的 student + KL 输出：

```bash
--skip-teacher-continuations
```

如果 KL 已经计算完成，可以只跑新增阶段，并用 `--gpu-ids "0 1"`、`--gpu-ids "7 8"` 或四张卡并行。完整命令、输出说明和故障排查见 [Teacher Top-KL 续写实验](teacher_kl_spike_continuations_zh.md)。
