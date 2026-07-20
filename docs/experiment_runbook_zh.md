# OPSD 训练、KL 对比与 Teacher 续写实验终端手册

本文档给出当前 `main` 上三类实验的完整终端步骤：

1. reference / semantic-skeleton OPSD 训练；
2. student rollout 上的 reference/skeleton teacher KL 对比；
3. 在全局 Top-KL 位置分别让 reference teacher 和 skeleton teacher 续写。

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

## 3. 一次跑完 rollout、KL 对比和 teacher 续写

这个入口默认依次运行：

1. 生成 student rollout；
2. 沿 student completion tokens 计算 `teacher_reference_vs_student` 和 `teacher_skeleton_vs_student` KL；
3. 选择全局 Top-KL token 位置，让两种 teacher 分别 greedy 续写；
4. 生成 student/reference/skeleton 三列 HTML 报告。

### LoRA/PEFT checkpoint，两张 GPU 4、5

```bash
KL_OUT=/path/to/outputs/student_teacher_kl_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /path/to/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --skeleton-file /path/to/full-train-skeletons.jsonl \
  --out "$KL_OUT" \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 20 \
  --seed 0
```

如果 checkpoint 是已经合并 adapter 的完整模型，删除 `--checkpoint-dir`，并把该模型目录直接传给 `--base-model`。

### TM-on student

将上面的：

```bash
--student-tm off
```

改成：

```bash
--student-tm on \
--max-new-tokens 16384 \
--max-model-len 32768
```

## 4. 只跑 rollout + KL 对比，不做续写

使用 `--skip-teacher-continuations`。建议保留 `$KL_OUT`，之后可以从该目录单独恢复续写阶段。

```bash
KL_OUT=/path/to/outputs/student_teacher_kl_only_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /path/to/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --skeleton-file /path/to/full-train-skeletons.jsonl \
  --out "$KL_OUT" \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --seed 0 \
  --skip-teacher-continuations
```

KL 对比完成后重点查看：

```text
$KL_OUT/student_rollouts.jsonl
$KL_OUT/student_teacher_category_kl.jsonl
$KL_OUT/student_teacher_category_kl_summary.json
```

summary 中主要比较：

- `teacher_reference_vs_student`；
- `teacher_skeleton_vs_student`；
- 两个 contrast 下的 `mean_style_kl`、`mean_math_kl`、`mean_other_kl`；
- `mean_style_kl_share`、`mean_math_kl_share`、`mean_other_kl_share`。

## 5. 从已有 KL 结果单独运行或恢复 teacher 续写

### 前置文件

`--out` 指向的 KL 结果目录至少需要：

```text
student_rollouts.jsonl
skeletons.jsonl
student_teacher_category_kl_shard*.jsonl
```

续写必须使用计算 KL 时相同的 base model 和 checkpoint。

### 两张 GPU 4、5

```bash
KL_OUT=/path/to/completed-kl-output

bash scripts/run_teacher_spike_continuations.sh \
  --base-model /path/to/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --out "$KL_OUT" \
  --gpu-ids "4 5" \
  --top-n 10 \
  --max-new-tokens 20 \
  --max-model-len 20000 \
  --hf-device-map cuda
```

续写阶段可以使用与 KL 阶段不同的 GPU 数量或编号；全局 Top-N 在分片前确定，因此改变 worker 数量不会改变入选位置。例如四张 GPU：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --base-model /path/to/Qwen3-1.7B \
  --checkpoint-dir /path/to/checkpoint-100 \
  --out /path/to/completed-kl-output \
  --gpu-ids "0 1 2 3" \
  --top-n 10 \
  --max-new-tokens 20
```

### 中断后重跑

如果一体化脚本已经完成 rollout 和 KL、但在 teacher continuation 阶段中断，不需要重跑 rollout/KL。直接重新执行本节的 `run_teacher_spike_continuations.sh` 命令即可。脚本会：

1. 从完整 KL shards 重新校验并原子生成 `student_teacher_category_kl_remerged.jsonl`；
2. 重新生成 continuation shards；
3. 原子合并并按全局 `rank` 排序；
4. 重建 summary 和 HTML 报告。

## 6. Teacher 续写输出

续写完成后查看：

```text
$KL_OUT/student_teacher_category_kl_remerged.jsonl
$KL_OUT/teacher_spike_continuations.jsonl
$KL_OUT/teacher_spike_continuation_summary.json
$KL_OUT/visualizations/teacher_spike_continuations.html
```

HTML 报告按高 KL 位置展示三列：

1. student 原始后续；
2. reference teacher 续写；
3. semantic-skeleton teacher 续写。

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
