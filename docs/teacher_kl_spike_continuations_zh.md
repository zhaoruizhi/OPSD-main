# Teacher 在 Top-KL 位置的续写实验

本文档说明如何在 student rollout 的全局高 KL 位置，让 reference teacher 和 skeleton teacher 分别继续生成少量 token，并把三条路径并排展示：

1. student 原始后续；
2. 看过完整 reference solution 的 teacher 后续；
3. 看过 semantic skeleton 的 teacher 后续。

默认选全局 Top 10 唯一位置，每条路径展示或生成 20 tokens，teacher 使用 greedy decoding。

## 分叉位置的准确含义

如果 KL 记录中的位置是 `p`，student completion token IDs 是：

```text
[t0, t1, ..., t(p-1), tp, t(p+1), ...]
```

那么本实验构造：

```text
teacher input  = privileged teacher prompt + [t0, ..., t(p-1)]
student 对照  = [tp, t(p+1), ..., t(p+19)]
teacher 输出  = 从 tp 之前开始 greedy 生成最多 20 tokens
```

所以 teacher 没有看到 student 在高 KL 位置已经选出的 `tp`。这正是观察 teacher 想用什么内容替换 student 路径所需要的分叉方式。

## 环境与输入要求

在仓库根目录运行命令。已有结果目录至少应包含：

```text
student_rollouts.jsonl
skeletons.jsonl
student_teacher_category_kl_shard*.jsonl
```

模型参数必须和计算 KL 时一致：

- KL 使用 base model 时，只传同一个 `--base-model`。
- KL 使用 LoRA/PEFT checkpoint 时，同时传相同的 `--base-model` 和 `--checkpoint-dir`。
- 不要用另一个 checkpoint 生成 continuation，否则续写与已保存的 KL 分布不是同一个模型。

## 直接复用当前结果：推荐命令

当前结果目录可以直接作为 `--out`。脚本会优先读取所有完整的 KL shard，并重新生成：

```text
student_teacher_category_kl_remerged.jsonl
```

因此不会依赖已经截断的 `student_teacher_category_kl.jsonl`，也不会覆盖它。

### 使用 GPU 0、1 两张卡

如果 KL 使用了 LoRA checkpoint：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --out /Users/zhaoruizhi/Desktop/code/OPSD-main/outputs/opsd_quick/qwen31b_tmon_student_teacher_category_kl_20260709_175323 \
  --base-model /path/to/Qwen3-model \
  --checkpoint-dir /path/to/checkpoint \
  --gpu-ids "0 1" \
  --top-n 10 \
  --max-new-tokens 20 \
  --max-model-len 20000 \
  --hf-device-map cuda
```

如果 KL 直接使用 base model，删除 `--checkpoint-dir`：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/qwen31b_tmon_student_teacher_category_kl_20260709_175323 \
  --base-model /path/to/Qwen3-model \
  --gpu-ids "0 1" \
  --top-n 10 \
  --max-new-tokens 20
```

### 使用 GPU 7、8 两张卡

只要服务器确实存在编号 7 和 8，就可以直接传入：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/result \
  --base-model /path/to/Qwen3-model \
  --checkpoint-dir /path/to/checkpoint \
  --gpu-ids "7 8" \
  --top-n 10 \
  --max-new-tokens 20
```

每个 Python worker 只会看到分配给自己的物理 GPU。例如物理 GPU 7 对应 worker 内部的 `cuda:0`。

### 使用四张卡

可以指定任意四个编号，例如 GPU 0、1、2、3：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/result \
  --base-model /path/to/Qwen3-model \
  --checkpoint-dir /path/to/checkpoint \
  --gpu-ids "0 1 2 3" \
  --top-n 10 \
  --max-new-tokens 20
```

也可以使用 GPU 4、5、6、7：

```bash
bash scripts/run_teacher_spike_continuations.sh \
  --out /path/to/result \
  --base-model /path/to/Qwen3-model \
  --checkpoint-dir /path/to/checkpoint \
  --gpu-ids "4 5 6 7" \
  --top-n 10 \
  --max-new-tokens 20
```

Top 10 在分片前已经全局确定。改变 GPU 数量只会改变任务分配，不会改变最终入选位置。四卡运行时 rank 采用取模分片，10 个位置大致分成 `3/3/2/2`。

## 从头运行完整 category-KL + continuation 实验

`scripts/run_student_teacher_category_kl.sh` 现在默认包含三个主要阶段：

1. student rollout；
2. reference/skeleton teacher vs student KL；
3. 全局 Top-KL teacher continuation 和 HTML 报告。

### TM-on、两张 GPU

```bash
BASE_MODEL=/path/to/Qwen3-model \
CHECKPOINT_DIR=/path/to/checkpoint \
SKELETON_FILE=/path/to/skeletons.jsonl \
OUT=/path/to/outputs/qwen31b_tmon_teacher_kl_continuation_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_student_teacher_category_kl.sh \
  --student-tm on \
  --sample-size 10 \
  --gpu-ids "0 1" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 20 \
  --seed 0
```

### TM-off、四张 GPU

```bash
BASE_MODEL=/path/to/Qwen3-model \
CHECKPOINT_DIR=/path/to/checkpoint \
SKELETON_FILE=/path/to/skeletons.jsonl \
OUT=/path/to/outputs/qwen31b_tmoff_teacher_kl_continuation_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_student_teacher_category_kl.sh \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5 6 7" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 20 \
  --seed 0
```

如果只想运行原来的 student + KL 流程，不生成 continuation：

```bash
bash scripts/run_student_teacher_category_kl.sh \
  --base-model /path/to/Qwen3-model \
  --skeleton-file /path/to/skeletons.jsonl \
  --gpu-ids "0 1" \
  --skip-teacher-continuations
```

## 输出文件

实验完成后重点查看：

```text
$OUT/student_teacher_category_kl_remerged.jsonl
$OUT/teacher_spike_continuation_shard<gpu>.jsonl
$OUT/teacher_spike_continuations.jsonl
$OUT/teacher_spike_continuation_summary.json
$OUT/visualizations/teacher_spike_continuations.html
```

其中：

- `student_teacher_category_kl_remerged.jsonl` 是从完整 shards 原子重建并逐行校验的 KL aggregate。
- `teacher_spike_continuations.jsonl` 是按 `rank=1..10` 排序后的完整记录。
- `teacher_spike_continuation_summary.json` 记录生成条件和实际 rank。
- `teacher_spike_continuations.html` 是最适合人工检查的三列报告。

快速检查结果数量：

```bash
python -m json.tool "$OUT/teacher_spike_continuation_summary.json"
wc -l "$OUT/teacher_spike_continuations.jsonl"
```

在本地文件系统中可以直接用浏览器打开 HTML；如果实验在远端服务器上，把 HTML 下载到本地再打开即可。

## HTML 报告内容

每个 Top-KL 位置显示：

- 全局 rank、problem ID、sample index、token position；
- `max KL`、reference KL、skeleton KL；
- 分叉前的 student 上下文和原 student token；
- Student original、Reference teacher、Skeleton teacher 三列 20-token 文本；
- 折叠显示的 problem、reference solution、semantic skeleton；
- 折叠显示的 teacher/student top-token distributions、delta log-prob 和 entropy。

如果 reference teacher 第一词是 `reference`，后面 20-token greedy continuation 会直接展示它是在说“according to the reference ...”，还是在引导某个具体公式、对象或结论。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--gpu-ids` | `"4"` | 空格分隔的物理 GPU 编号，数量自动决定 worker 数 |
| `--top-n` | `10` | 全局唯一 KL 分叉位置数量 |
| `--max-new-tokens` | `20` | 每种 teacher greedy 续写的最大 token 数；student 也展示相同长度 |
| `--max-model-len` | `20000` | teacher prompt + student prefix + continuation 的最大总长度 |
| `--hf-device-map` | `cuda` | `cuda`、`auto` 或 `cpu`；服务器 GPU 实验建议 `cuda` |
| `--kl-file` | 自动寻找 shards | 手动指定 KL aggregate；脚本仍会验证并写入 remerged 文件 |
| `--student-rollout-file` | `$OUT/student_rollouts.jsonl` | 原 student rollout |
| `--skeleton-file` | `$OUT/skeletons.jsonl` | semantic skeleton 文件 |

完整流程对应参数名为：

- `--teacher-continuation-top-n`，默认 10；
- `--teacher-continuation-max-new-tokens`，默认 20；
- `--skip-teacher-continuations`，跳过 Phase 3。

## 常见报错

### JSONL 损坏

合并器会报告具体文件和行号，例如：

```text
Invalid JSONL at .../student_teacher_category_kl_shard7.jsonl:64
```

目标 aggregate 只会在所有输入均验证成功后原子替换，因此失败不会留下一个看似存在但内容不完整的新文件。

### 缺少 reference 或 skeleton contrast

Top 10 的每个位置都必须能连接到两种 contrast。如果缺少其中一种，脚本会停止，而不会把缺失 KL 当作 0。

### 上下文超过限制

脚本不会裁掉 privileged prompt 或 student prefix。出现 `exceeds max context` 时，应把 `--max-model-len` 调整到模型实际支持的长度，例如：

```bash
--max-model-len 32768
```

### GPU 编号无效

先在服务器检查：

```bash
nvidia-smi -L
```

然后把实际存在的编号以空格分隔传给 `--gpu-ids`。
