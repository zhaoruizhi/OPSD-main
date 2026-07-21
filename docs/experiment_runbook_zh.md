# OPSD 训练、KL 对比与 Teacher 续写实验终端手册

本文档给出当前 `main` 上三类实验的完整终端步骤：

1. reference / semantic-skeleton OPSD 训练；
2. 四路 rollout 的 performance / token length 与两套 KL 对比；
3. 在 student 轨迹的全局 Top-KL 位置分别让 reference teacher 和 skeleton teacher 续写。

所有命令都从仓库根目录执行：

```bash
cd /Users/zhaoruizhi/Desktop/code/OPSD-main
```

## 0. GPU 参数格式

训练和 KL runner 的 GPU 参数格式不同：

| 实验 | 参数 | 格式 | 示例 |
| --- | --- | --- | --- |
| OPSD 训练 | `TRAIN_GPU_IDS` | 逗号分隔 | `4,5` |
| KL 对比 / teacher 续写 | `--gpu-ids` | 空格分隔，整体加引号 | `"4 5"` |

开始前先确认物理 GPU 编号：

```bash
nvidia-smi -L
```

训练时 `NUM_PROCESSES` 通常等于 `TRAIN_GPU_IDS` 中的 GPU 数量。例如选 GPU 4、5，就使用 `TRAIN_GPU_IDS=4,5 NUM_PROCESSES=2`。同时启动多个训练时，每个训练必须使用不同的 `MAIN_PROCESS_PORT`。

KL 与续写 runner 会为 `--gpu-ids` 中的每个编号启动一个 worker，并通过 `CUDA_VISIBLE_DEVICES` 把 worker 限制到对应的物理 GPU；worker 内部看到的设备仍是 `cuda:0`。

## 1. Reference OPSD 训练

### 输入

- base model、输出目录和训练超参来自 `scripts/run_opsd_1b.sh`；
- 不需要 skeleton 文件；
- 首次使用 W&B 时先运行 `wandb login`。

### 两张 GPU 4、5

```bash
TRAIN_GPU_IDS=4,5 \
NUM_PROCESSES=2 \
MAIN_PROCESS_PORT=12949 \
bash scripts/run_opsd_1b.sh
```

### 四张 GPU 0、1、2、3

```bash
TRAIN_GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=12949 \
bash scripts/run_opsd_1b.sh
```

训练 run 名为：

```text
qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005
```

## 2. Semantic-skeleton OPSD 训练

### 输入检查

`SKELETON_FILE` 必须覆盖训练集完整 train split，并以 dataset row index 作为 `problem_id`：

```bash
SKELETON_FILE=/path/to/full-train-skeletons.jsonl
wc -l "$SKELETON_FILE"
```

正式训练脚本使用 `--skeleton_subset_policy error`；缺少任意训练样本的 skeleton 时会直接停止。

### 两张 GPU 4、5

```bash
SKELETON_FILE=/path/to/full-train-skeletons.jsonl \
TRAIN_GPU_IDS=4,5 \
NUM_PROCESSES=2 \
MAIN_PROCESS_PORT=12950 \
bash scripts/run_opsd_1b_skeleton.sh
```

### 四张 GPU 0、1、2、3

```bash
SKELETON_FILE=/path/to/full-train-skeletons.jsonl \
TRAIN_GPU_IDS=0,1,2,3 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=12950 \
bash scripts/run_opsd_1b_skeleton.sh
```

训练 run 名为：

```text
qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005
```

reference 和 skeleton 公平对比时使用相同 GPU 数量、训练超参和 checkpoint 步数，但用不同的 `MAIN_PROCESS_PORT` 并分别启动。

训练脚本支持两种 skeleton teacher prompt：

| `TEACHER_PROMPT_PROFILE` | skeleton 是否看见答案 | JSON 检查字段 |
| --- | --- | --- |
| `current-style-neutral`（默认） | 否 | `check` |
| `legacy-20260629` | 是，在 skeleton block 后有单独的 `Final answer:` | `checks` |

### 复刻 2026-06-29 prompt 的全量训练

KL 复刻目录中的 `skeletons.jsonl` 只有 128 条，不可用于全量训练。这里使用覆盖约 29.4k train split 的全量文件，同时只把已验证的旧 prompt 引入训练；训练 rollout 仍保持 `max_completion_length=1024`。

```bash
cd /home/ruizzhao/OPSD-main
git pull --ff-only origin main

export SKELETON_FILE=/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl
export TEACHER_PROMPT_PROFILE=legacy-20260629
export RUN_CONFIG=qwen31b_gen1024_skeleton_legacy20260629_fixteacher_temp11_forwardbeta0_clip005
export MODEL_NAME_OR_PATH=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B
export OUTPUT_DIR=/home/ruizzhao/OPSD-main/outputs/opsd/

export TRAIN_GPU_IDS=4,5,6,7
export NUM_PROCESSES=4
export MAIN_PROCESS_PORT=12950

test -f "$SKELETON_FILE"
wc -l "$SKELETON_FILE"
wandb status

bash scripts/run_opsd_1b_skeleton.sh
```

训练启动后确认终端打印 `Teacher Prompt Profile: legacy-20260629`，并确认完整训练行数没有被过滤。最终目录为：

```text
/home/ruizzhao/OPSD-main/outputs/opsd/qwen31b_gen1024_skeleton_legacy20260629_fixteacher_temp11_forwardbeta0_clip005/
```

rank 0 会在其中生成 `experiment_config.json`，记录解析后的全部 script/training/model 参数、命令行、GPU/分布式环境、Git commit/dirty 状态以及 skeleton 文件 SHA-256：

```bash
TRAIN_OUT="$OUTPUT_DIR/$RUN_CONFIG"
sed -n '1,240p' "$TRAIN_OUT/experiment_config.json"
```

## 3. 一次跑完四路 rollout、两套 KL 和 teacher 续写

`scripts/run_student_teacher_category_kl.sh` 现在依次运行：

1. 在同一批题上生成 `student`、`teacher_base`、`teacher_reference`、`teacher_skeleton` 四路 rollout；
2. 汇总四路 performance 和 `avg_completion_tokens`；
3. 在固定 `teacher_base` rollout 上计算 `teacher_reference_vs_teacher_base`、`teacher_skeleton_vs_teacher_base` KL，并计算四路 rollout entropy；
4. 生成旧实验同名的 teacher-base KL CSV、JSONL、HTML；
5. 在固定 student rollout 上计算 `teacher_reference_vs_student`、`teacher_skeleton_vs_student` 分类 KL；
6. 从 student 轨迹 KL 中选择全局 Top-KL token，让两种 teacher 分别 greedy 续写并生成三列 HTML。

两套 KL 使用相同模型权重，但 target trajectory 和 baseline 不同，结果文件不会相互覆盖。teacher prompt 由 `--experiment-profile` 明确选择：

| profile | 用途 | skeleton 是否看见答案 | KL target token source |
| --- | --- | --- | --- |
| `legacy-20260629` | 复刻归档实验 | 是，单独的 `Final answer:` 行 | `full_generation` 文本重新 tokenize |
| `current-style-neutral` | 当前 prompt 实验 | 否 | 优先复用 rollout 原始 token IDs |

每次运行都会在 `$OUT/experiment_config.json` 保存 profile、Git commit、输入文件 SHA-256、GPU IDs、`n` 和所有主要 sampling/token 参数。

### Rollout 长度参数（重要）

`--max-model-len` 不是生成长度。它是模型可使用的总上下文上限，即 prompt tokens 与 completion tokens 的总和。四路 rollout 的生成长度由下面两个参数分别控制：

| 参数 | 控制范围 | 当前默认值 |
| --- | --- | --- |
| `--student-max-new-tokens` | 只控制 `student` | TM-off 为 `1024`；TM-on 为 `16384` |
| `--teacher-max-new-tokens` | 控制 `teacher_base`、`teacher_reference`、`teacher_skeleton` | `16384` |
| `--max-model-len` | prompt + completion 的总上下文 | `20000` |

一键 runner 的 `--val-n` 默认值现在是 `4`。`legacy-20260629` 在没有显式覆盖时还会采用 `sample-size=128` 和 student/teacher `max-new-tokens=16384`，并强制要求传入归档 `sample_indices.json`。

兼容旧命令的 `--max-new-tokens N` 仍然保留，它会同时设置 student 和三路 teacher；如果同时传入分组参数，则 `--student-max-new-tokens` 或 `--teacher-max-new-tokens` 优先。正式对比建议显式写出两个分组参数，避免把 context length 误当成 completion length。

### 复刻 2026-06-29 归档实验：Qwen3-1.7B、GPU 4/5/6/7

```bash
cd /home/ruizzhao/OPSD-main

LEGACY_DIR=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_skeleton_ablation_reuse_20260629_112333
KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/legacy_20260629_reproduction_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --experiment-profile legacy-20260629 \
  --sample-indices-file "$LEGACY_DIR/sample_indices.json" \
  --skeleton-file "$LEGACY_DIR/skeletons.jsonl" \
  --out "$KL_OUT" \
  --student-tm off \
  --student-max-new-tokens 16384 \
  --teacher-max-new-tokens 16384 \
  --sample-size 128 \
  --val-n 4 \
  --temperature 1.1 \
  --top-p 0.95 \
  --top-k 20 \
  --gpu-memory-utilization 0.9 \
  --gpu-ids "4 5 6 7" \
  --max-model-len 20000 \
  --trajectory-sample-index 0 \
  --probe-tokens 0 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 200 \
  --seed 0
```

归档结果中 teacher 三路存在恰好 16384 tokens 且 `finish_reason=length` 的记录，因此这里以实际产物为准显式使用 16384；旧提交脚本中的 1024 默认值和文档中的 8194 示例都不能解释该产物。

### 当前 style-neutral prompt 的受控对照

为只比较 rollout performance 的 prompt 差异，复用完全相同的旧 manifest、旧 skeleton 和其他参数，将上面命令中的 profile 改为：

```bash
--experiment-profile current-style-neutral
```

如果 KL 也要只隔离 prompt、保持旧文本重分词口径，再增加：

```bash
--target-token-source target_tail_text
```

不显式覆盖时，current profile 的 KL 使用原始 rollout token IDs，legacy profile 使用旧文本重分词口径。两次运行的 `$OUT/experiment_config.json` 会明确记录该差异。

如果测试 LoRA/PEFT checkpoint，在命令中增加：

```bash
--checkpoint-dir /path/to/checkpoint-100
```

如果 checkpoint 已经合并成完整模型，不传 `--checkpoint-dir`，把合并后的模型目录直接传给 `--base-model`。

### TM-on student

teacher 三路保持当前 thinking 设置；只把 student 改为 TM-on：

```bash
--student-tm on \
--student-max-new-tokens 16384 \
--teacher-max-new-tokens 16384 \
--max-model-len 32768
```

## 4. 只跑四路 rollout + 两套 KL，不做续写

在上一节命令末尾增加 `--skip-teacher-continuations`：

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_only_$(date +%Y%m%d_%H%M%S)

LEGACY_DIR=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_skeleton_ablation_reuse_20260629_112333

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --experiment-profile legacy-20260629 \
  --sample-indices-file "$LEGACY_DIR/sample_indices.json" \
  --skeleton-file "$LEGACY_DIR/skeletons.jsonl" \
  --out "$KL_OUT" \
  --student-tm off \
  --student-max-new-tokens 16384 \
  --teacher-max-new-tokens 16384 \
  --sample-size 128 \
  --val-n 4 \
  --gpu-ids "4 5 6 7" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --seed 0 \
  --skip-teacher-continuations
```

### Performance 和 token length

查看：

```text
$KL_OUT/rollout_summary.json
```

`conditions` 下有四组：

- `student`
- `teacher_base`
- `teacher_reference`
- `teacher_skeleton`

每组的 performance 字段包括 `avg_at_n`、`pass_at_n`、`majority_vote`、`format_rate`；token length 字段是 `avg_completion_tokens`。如果主要比较 student/reference/skeleton 三路，直接从这三组读取即可；teacher_base 作为旧 KL 基准一并保留。

### Teacher-base 轨迹 KL（旧实验口径）

```text
$KL_OUT/logit_probe.jsonl
$KL_OUT/logit_summary.json
$KL_OUT/visualizations/teacher_base_kl_reference_vs_skeleton_report.html
$KL_OUT/visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv
$KL_OUT/visualizations/teacher_base_top_distribution_spikes.jsonl
```

`logit_summary.json` 中比较：

- `teacher_reference_vs_teacher_base`
- `teacher_skeleton_vs_teacher_base`

同时包含 `student`、`teacher_base`、`teacher_reference`、`teacher_skeleton` 四路 `rollout_entropy`。

### 只修复已有结果的 KL 可视化（不使用 GPU）

如果 rollout 和 KL 已经完成，但 HTML 中曲线、热力图或 spike 表为空，可以直接根据现有 JSONL 重建三个可视化文件：

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_YYYYMMDD_HHMMSS

python3 eval/quick_teacher_base_kl_report.py \
  --logit-file "$KL_OUT/logit_probe.jsonl" \
  --rollout-file "$KL_OUT/rollouts.jsonl" \
  --rollout-summary-file "$KL_OUT/rollout_summary.json" \
  --skeleton-file "$KL_OUT/skeletons.jsonl" \
  --csv-file "$KL_OUT/visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv" \
  --spikes-jsonl-file "$KL_OUT/visualizations/teacher_base_top_distribution_spikes.jsonl" \
  --report-file "$KL_OUT/visualizations/teacher_base_kl_reference_vs_skeleton_report.html"
```

这条命令只修复可视化。已经在 1024 tokens 处被截断的 teacher rollout 无法从现有文件补回，因此 performance 和 token length 必须使用修复后的 runner 重新跑。

### Student 轨迹 KL（分类与续写口径）

```text
$KL_OUT/student_teacher_category_kl.jsonl
$KL_OUT/student_teacher_category_kl_summary.json
```

其中比较：

- `teacher_reference_vs_student`
- `teacher_skeleton_vs_student`

rollout 始终保留 completion token IDs；KL 记录的 `target_token_source` 则由 profile/显式参数决定，并用于后续 Top-KL teacher 续写的位置对齐。分类字段包括 `mean_style_kl`、`mean_math_kl`、`mean_other_kl` 及对应 KL share。

## 5. 从已有双 KL 结果单独运行或恢复 teacher 续写

`$KL_OUT` 至少需要：

```text
rollouts.jsonl
skeletons.jsonl
student_teacher_category_kl_shard*.jsonl
```

续写必须使用计算 KL 时相同的 base model 和 checkpoint。完整服务器命令：

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_YYYYMMDD_HHMMSS

bash scripts/run_teacher_spike_continuations.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --out "$KL_OUT" \
  --student-rollout-file "$KL_OUT/rollouts.jsonl" \
  --skeleton-file "$KL_OUT/skeletons.jsonl" \
  --teacher-prompt-profile legacy-20260629 \
  --gpu-ids "4 5" \
  --top-n 10 \
  --max-new-tokens 200 \
  --max-model-len 20000 \
  --hf-device-map cuda
```

如果使用 adapter，增加同一个 `--checkpoint-dir /path/to/checkpoint-100`。续写阶段可以更换 GPU 数量或编号；全局 Top-N 在分片前确定，因此 worker 数量不会改变入选位置。

如果一体化脚本在 continuation 阶段中断，不需要重跑 rollout 或两套 KL。上面的命令会：

1. 从 student-trajectory KL shards 校验并原子生成 `student_teacher_category_kl_remerged.jsonl`；
2. 重新生成 continuation shards；
3. 原子合并并按全局 `rank` 排序；
4. 重建 summary 和 HTML 报告。

## 6. Teacher 续写输出

```text
$KL_OUT/student_teacher_category_kl_remerged.jsonl
$KL_OUT/teacher_spike_continuations.jsonl
$KL_OUT/teacher_spike_continuation_summary.json
$KL_OUT/visualizations/teacher_spike_continuations.html
```

HTML 按高 KL 位置展示 student 原始后续、reference teacher 续写、semantic-skeleton teacher 续写三列。

## 7. 常见启动问题

### GPU 数量与训练进程数不一致

如果训练使用 `TRAIN_GPU_IDS=4,5`，应同时设置 `NUM_PROCESSES=2`。不要把训练格式写成 `"4 5"`。

### KL runner 只启动一个 worker

KL 参数必须整体加引号并以空格分隔：

```bash
--gpu-ids "4 5"
```

不要写成训练使用的逗号格式 `4,5`。

### 并行训练端口冲突

同时跑 reference 和 skeleton 时设置不同端口，例如 `12949` 和 `12950`。

### Continuation 找不到 KL 输入

确认 `$KL_OUT` 中存在 `student_teacher_category_kl_shard*.jsonl`。如果只有一个手动生成的 aggregate，也可以向续写脚本显式传：

```bash
--kl-file /path/to/student_teacher_category_kl.jsonl
```

### 传了 `--max-model-len 20000`，结果仍显示 1024 tokens

`20000` 是 prompt + completion 的总上下文上限，不是要求模型生成 20000 tokens。检查启动日志中的三行：

```text
Student TM: off | Student max new tokens: 1024
Teacher max new tokens: 16384
Model context length: 20000
```

如果 teacher rollout 的 `finish_reason` 大量为 `length` 且 token 数等于 16384，说明 teacher 触及的是生成上限；此时再结合显存和 prompt 长度决定是否提高 `--teacher-max-new-tokens` 与 `--max-model-len`。
