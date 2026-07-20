# 四路 Rollout、双 KL 与 Teacher 续写说明

从 OPSD 训练到实验执行的完整命令总表见 [OPSD 实验终端手册](experiment_runbook_zh.md)。

## 实验现在计算什么

`scripts/run_student_teacher_category_kl.sh` 对同一批 problem 先生成四路 rollout：

| condition | prompt | thinking |
| --- | --- | --- |
| `student` | 只包含 problem | 由 `--student-tm off/on` 控制 |
| `teacher_base` | 只包含 problem | 开启 |
| `teacher_reference` | 当前 reference teacher prompt | 开启 |
| `teacher_skeleton` | 当前 style-neutral semantic skeleton prompt | 开启 |

这里的 teacher 是“同一套模型权重在不同 privileged prompt 下的条件分布”。如果传入 LoRA checkpoint，四路 rollout、两套 KL 和续写都会使用同一个 base model + adapter。

当前 skeleton teacher prompt 保持为：

```text
Problem: {problem}
Below is a style-neutral semantic skeleton extracted from a reference solution.
=== Semantic Skeleton Begin ===
{skeleton}
=== Semantic Skeleton End ===
...
```

不会回退到 `Here is a reference solution to this problem`，也不会向 skeleton teacher 注入最终答案。

## Performance 和 token length

四路 rollout 合并到：

```text
$OUT/rollouts.jsonl
```

整体汇总写到：

```text
$OUT/rollout_summary.json
```

`conditions` 下分别有：

- `student`
- `teacher_base`
- `teacher_reference`
- `teacher_skeleton`

每组主要字段：

| 字段 | 含义 |
| --- | --- |
| `avg_at_n` | 所有生成的平均正确率 |
| `pass_at_n` | 每题至少一个生成正确的比例 |
| `majority_vote` | 每题多数答案正确的比例 |
| `format_rate` | 成功提取 boxed answer 的比例 |
| `avg_completion_tokens` | 平均 completion token length |

默认 `VAL_N=1`，所以当前一键脚本中 `avg_at_n`、`pass_at_n` 和单样本正确率口径一致。如果只需要 student/reference/skeleton 三路 token length，读取这三组的 `avg_completion_tokens`；`teacher_base` 同时保留，供旧 KL 口径使用。

## 第一套 KL：teacher_base 轨迹

固定 target trajectory 为 `teacher_base` 自己生成并保存的 `completion_token_ids`，计算：

```text
KL(P_teacher_reference || P_teacher_base)
KL(P_teacher_skeleton  || P_teacher_base)
```

输出：

```text
$OUT/logit_probe_shard*.jsonl
$OUT/logit_probe.jsonl
$OUT/logit_summary_shard*.json
$OUT/logit_summary.json
```

`logit_summary.json` 的 contrast 名称是：

- `teacher_reference_vs_teacher_base`
- `teacher_skeleton_vs_teacher_base`

这套 probe 不传 `--skip-rollout-entropy`，所以同一个 summary 中还有四路 `rollout_entropy`。

旧实验同名可视化产物也会自动生成：

```text
$OUT/visualizations/teacher_base_kl_reference_vs_skeleton_report.html
$OUT/visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv
$OUT/visualizations/teacher_base_top_distribution_spikes.jsonl
```

HTML 顶部直接显示四路 performance 和 `avg_completion_tokens`，下方展示 teacher-base token 轨迹上的 reference/skeleton KL 曲线、token heatmap 和 top distributions。

## 第二套 KL：student 轨迹

固定 target trajectory 为 student 保存的 `completion_token_ids`，计算：

```text
KL(P_teacher_reference || P_student)
KL(P_teacher_skeleton  || P_student)
```

输出：

```text
$OUT/student_teacher_category_kl_shard*.jsonl
$OUT/student_teacher_category_kl.jsonl
$OUT/student_teacher_category_kl_summary_shard*.json
$OUT/student_teacher_category_kl_summary.json
```

contrast 名称是：

- `teacher_reference_vs_student`
- `teacher_skeleton_vs_student`

这套 KL 继续提供 token category 汇总：

- `mean_style_kl`
- `mean_math_kl`
- `mean_other_kl`
- `mean_style_kl_share`
- `mean_math_kl_share`
- `mean_other_kl_share`
- `style_token_count`
- `math_token_count`
- `other_token_count`

teacher 续写只从这套 student-trajectory KL 选择 Top-KL 位置，不会读取 `logit_probe.jsonl`。

## 一次跑完：两套 KL + 续写

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --skeleton-file /home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl \
  --out "$KL_OUT" \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --teacher-continuation-top-n 10 \
  --teacher-continuation-max-new-tokens 200 \
  --seed 0
```

如果评测 LoRA checkpoint，增加：

```bash
--checkpoint-dir /path/to/checkpoint-100
```

## 只跑四路 rollout + 两套 KL

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_only_$(date +%Y%m%d_%H%M%S)

bash scripts/run_student_teacher_category_kl.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --skeleton-file /home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl \
  --out "$KL_OUT" \
  --student-tm off \
  --sample-size 10 \
  --gpu-ids "4 5" \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --seed 0 \
  --skip-teacher-continuations
```

这条命令仍然生成 performance、token length、两套 KL 和 teacher-base KL 可视化；只跳过生成式 teacher 续写。

## 从已有结果恢复续写

```bash
cd /home/ruizzhao/OPSD-main

KL_OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/student_teacher_dual_kl_YYYYMMDD_HHMMSS

bash scripts/run_teacher_spike_continuations.sh \
  --base-model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --out "$KL_OUT" \
  --student-rollout-file "$KL_OUT/rollouts.jsonl" \
  --skeleton-file "$KL_OUT/skeletons.jsonl" \
  --gpu-ids "4 5" \
  --top-n 10 \
  --max-new-tokens 200 \
  --max-model-len 20000 \
  --hf-device-map cuda
```

续写产物：

```text
$OUT/student_teacher_category_kl_remerged.jsonl
$OUT/teacher_spike_continuation_shard*.jsonl
$OUT/teacher_spike_continuations.jsonl
$OUT/teacher_spike_continuation_summary.json
$OUT/visualizations/teacher_spike_continuations.html
```

续写阶段先从 student KL shards 原子重建并校验 aggregate，再选择全局 Top-N。因此可以在 KL 完成后换用不同的 GPU 编号或数量。

## GPU 分配

```bash
--gpu-ids "4 5"
```

会为 GPU 4、5 各启动一个 shard worker。四路 rollout、teacher-base KL、student KL 和续写阶段都只使用这两个物理 GPU。worker 进程内部看到的设备是 `cuda:0`，对应外部 `CUDA_VISIBLE_DEVICES` 分配的物理卡。

GPU ID 用空格分隔并整体加引号；不要写成训练脚本使用的逗号格式 `4,5`。

## 常用快速参数

- `--sample-size 10`：同一份 sample manifest 用于四路 rollout。
- `--sample-indices-file /path/to/sample_indices.json`：复用固定题目。
- `--probe-tokens 128`：只 probe 每条 response 的前 128 tokens，用于 smoke test。
- `--probe-tokens 0`：probe 完整 response，正式实验使用。
- `--trajectory-sample-index 0`：从每个 condition 选择 `sample_index=0` 做两套固定轨迹 KL。
- `--student-tm on`：只改变 student condition；建议同时提高 `--max-new-tokens` 和 `--max-model-len`。
- `--skip-teacher-continuations`：停止在两套 KL 与 teacher-base 报告之后。

## 结果完整性检查

```bash
python3 - <<'PY'
import json
from pathlib import Path

out = Path("/path/to/student_teacher_dual_kl_output")
rollout = json.loads((out / "rollout_summary.json").read_text())
base_kl = json.loads((out / "logit_summary.json").read_text())
student_kl = json.loads((out / "student_teacher_category_kl_summary.json").read_text())

print("rollout conditions:", sorted(rollout["conditions"]))
for condition, metrics in rollout["conditions"].items():
    print(condition, "performance=", metrics["avg_at_n"], "tokens=", metrics["avg_completion_tokens"])
print("teacher-base KL:", sorted(base_kl["contrasts"]))
print("entropy:", sorted(base_kl["rollout_entropy"]))
print("student KL:", sorted(student_kl["contrasts"]))
PY
```

预期 condition 和 entropy 都包含四路；teacher-base KL 与 student KL 各包含两个正确命名的 contrast。
