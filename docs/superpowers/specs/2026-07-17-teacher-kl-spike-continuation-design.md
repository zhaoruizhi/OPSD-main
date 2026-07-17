# Teacher KL Spike Continuation 设计

## 目标

在现有 student rollout 上完成 reference teacher / skeleton teacher 与 student 的逐 token KL 对比后，选出全局 KL 最大的 10 个唯一分叉位置。对每个位置使用完全相同的 student 前缀，让 reference teacher 和 skeleton teacher 分别以 greedy decoding 续写约 20 tokens，并与 student 原始后续 20 tokens 并排展示，从而观察特权信息具体想把 student 引向什么内容。

## 已确认的实验语义

- KL 记录中的 position `p` 表示模型在看到 student completion 的 `tokens[:p]` 后，对第 `p` 个 token 的预测分布。
- teacher continuation 必须从 `tokens[:p]` 分叉，不能先喂入 student 的第 `p` 个 token；否则无法观察 teacher 对该高 KL token 的替代意图。
- Top 10 是所有 problem、sample 和 reference/skeleton contrast 合并后的全局 Top 10 唯一位置，而不是每个 shard 或每道题各取 10 个。
- 同一位置即使同时出现在 reference 和 skeleton contrast 中也只展示一次。
- 排名分数取该位置的 `max(reference_kl, skeleton_kl)`。
- reference teacher 与 skeleton teacher 都使用 greedy decoding，默认 `max_new_tokens=20`，避免随机采样掩盖 teacher 的最高概率路径。
- student 对照文本直接取原 student completion 的 `tokens[p:p+20]`，不重新生成。
- 生成必须使用与 KL probe 相同的 base model、可选 LoRA checkpoint、chat template、thinking 设置和 privileged prompt 构造逻辑。

## 方案比较

### 方案 A：KL 计算时在每个 shard 内直接续写

优点是少一次后处理入口。缺点是每个进程只能看到自己的局部 KL，无法在不增加跨进程同步的情况下得到真正的全局 Top 10；而且 KL 完成前无法知道最终候选位置。

### 方案 B：KL 完成后独立执行全局 spike continuation（采用）

先原子合并并校验 KL shards，再进行全局选点，然后把最终 10 个位置按 rank 均匀分配给用户指定的 GPU。该方案既能复用已有结果，也能保证全局排名正确，并支持 1、2、4 或任意数量的 GPU 编号。

### 方案 C：只提供单卡离线 continuation 脚本

实现最简单，但无法满足在不同机器上指定 2 或 4 张 GPU 并行运行的要求。

## 架构与组件

### 1. 原子 JSONL shard 合并器

新增一个小型命令行工具，逐行解析并校验每个 shard 的 JSON，写入同目录临时文件，所有输入均成功后再原子替换目标文件。它用于合并 student rollout、KL records 和 continuation records。

该合并方式解决当前结果目录中的实际问题：`student_teacher_category_kl.jsonl` 只有 125 条完整记录和 1 条截断记录，而四个 shard 文件各有 64 条完整记录。后续实验应从 shard 重建完整 aggregate，不能在损坏文件上选择 Top 10。

### 2. 全局 spike 选择器

新增 teacher spike continuation probe，流式读取 KL JSONL：

1. 对每条 contrast record 的完整 `kl_per_token` 取局部 Top N 候选。
2. 按 `(problem_id, sample_index, target_condition, position)` 去重。
3. 以 reference/skeleton KL 的较大值做全局排序，保留 Top 10。
4. 第二次读取 KL 文件，为最终位置补齐两种 contrast 的 KL、delta log-prob、entropy 和已保存的 top-token distribution。
5. 使用 `(problem_id, sample_index, target_condition)` 精确连接 student rollout，并校验 position 未超出 `completion_token_ids`。

选择逻辑直接使用完整的 `kl_per_token`，而不是依赖 HTML 或仅依赖每条 record 已保存的 `top_kl_positions`，避免漏掉真正的全局高值。

### 3. teacher greedy continuation

对每个分叉位置构建：

```text
reference input = reference privileged prompt + student completion tokens[:p]
skeleton input  = skeleton privileged prompt  + student completion tokens[:p]
student display = student completion tokens[p:p+20]
```

teacher prompt 复用 `eval/quick_logit_probe.py` 中的 `context_prompt_ids_for_condition()`，确保与 KL probe 的条件完全一致。生成使用 HuggingFace causal LM 和可选 PEFT adapter；每个 worker 只看一张由 `CUDA_VISIBLE_DEVICES` 映射的 GPU。

若 `prompt_tokens + prefix_tokens + max_new_tokens > max_context_tokens`，记录明确错误并停止该 worker，不允许静默裁掉 privileged prompt 或 student prefix，因为那会改变被测条件。

### 4. 多 GPU 调度

新增独立 shell 入口用于已有 KL 结果，并把同一阶段集成到完整 category-KL 脚本：

- `--gpu-ids "0 1"` 使用两张卡。
- `--gpu-ids "0 1 2 3"` 使用四张卡。
- `--gpu-ids "7 8"` 使用编号 7、8 的两张卡（前提是机器确实暴露这些编号）。
- GPU 数量由传入编号的数量自动推导；每个 worker 接收 `--shard-id` 和 `--num-shards`。
- Top 10 先全局确定，再按 rank 分片，所以增加或减少 GPU 不会改变入选位置。

完整 category-KL 脚本新增：

- `--teacher-continuation-top-n`，默认 `10`。
- `--teacher-continuation-max-new-tokens`，默认 `20`。
- `--skip-teacher-continuations`，需要只跑原 KL 流程时使用。

独立 continuation 脚本提供相同的 Top N、token 数、model、checkpoint、GPU 和 context length 参数，便于直接复用当前输出目录。

## 输出

新阶段在实验目录生成：

- `teacher_spike_continuations.jsonl`：按全局 rank 排序的 10 条完整记录。
- `teacher_spike_continuation_summary.json`：配置、记录数、成功/失败数和输入来源。
- `visualizations/teacher_spike_continuations.html`：三列并排报告。

每条 JSONL 至少包含：

- rank、problem/sample、position、student token、max/reference/skeleton KL；
- 分叉前上下文 snippet；
- student 原始 20-token 后缀；
- reference teacher 20-token continuation；
- skeleton teacher 20-token continuation；
- 两种 teacher 的 prompt token 数、prefix token 数、生成 token IDs、文本和停止原因；
- problem、reference solution 和 semantic skeleton，供 HTML 折叠查看；
- 两种 contrast 在该位置的 top-token distributions（若 KL 文件已保存）。

HTML 每个 rank 使用一个区块，顶部显示 problem、位置和 KL；中部高亮分叉点；下方三列分别显示 student、reference teacher、skeleton teacher。reference solution、skeleton 和 top distributions 默认折叠，避免主视图过长。

## 失败处理与可恢复性

- 任一输入 JSONL 存在损坏行时，合并/选择立即失败，并报告文件和行号。
- shard 合并使用临时文件和原子替换；失败不会覆盖已有 aggregate。
- 缺少 rollout、skeleton、adapter 文件或对应 problem/sample 时立即失败并给出具体 key。
- worker 输出按 shard 独立保存；已完成 shard 可以复用，重新合并不会改变 rank。
- 不修改当前损坏的 aggregate；实验命令显式从现有完整 shard 重建一个有效 aggregate 后再运行。

## 测试与验收

自动化测试覆盖：

- 多 contrast 同一 position 去重以及 `max(reference_kl, skeleton_kl)` 全局排序；
- continuation 输入严格排除 student 的第 `p` 个 token；
- student 对照从第 `p` 个 token 开始且最多 20 tokens；
- reference/skeleton prompt 使用正确 privileged condition；
- rank 在 2/4 个 shard 间稳定、无重复、无遗漏；
- JSONL 损坏检测和原子合并；
- HTML 包含三列内容并正确转义模型文本；
- shell 参数支持 `--gpu-ids`、Top N、20-token 默认值和 skip 开关。

验收时运行相关单元测试、shell 语法检查和现有 quick probe 回归测试。由于本地工作区没有服务器上的 Qwen checkpoint/GPU，真实模型生成需在实验机器上按文档命令执行；代码级生成路径用轻量 fake model/tokenizer 测试固定输入边界和输出结构。

