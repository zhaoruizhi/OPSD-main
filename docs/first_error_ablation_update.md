# First-Error Ablation 更新说明

## 更新概览

当前 `main` 已到远端最新提交 `9df5674 Add first-error ablation pipeline`。这次主要补上了一个新的 first-error ablation 实验流程，用来比较两种 teacher 信息对学生错误轨迹的修复效果：

- `teacher_base_w_text`: teacher 看到 final answer 和完整 reference solution。
- `teacher_base_w_first_error`: teacher 看到 final answer 和 first-error diagnostic JSON。

新增和修改的核心文件：

- `eval/quick_first_error_ablation.py`: 新增 first-error continuation 和 segmented KL 主逻辑。
- `scripts/run_first_error_ablation.sh`: 新增一键运行脚本，串起 sample、student rollout、first-error diagnostic、continuation、KL。
- `tests/test_first_error_ablation.py`: 新增 schema、prompt、token range、segment KL 单测。
- `generate_1st-error_json.py`: first-error 诊断 schema 改为 `first_error_sentence`，并支持基于 student sample-0 rollout 生成诊断。
- `eval/quick_opsd_common.py`: 新增 first-error diagnostic 校验、prompt 构造、文本切片和 token range 工具。

另一个需要注意的近期更新是 KL probe 已优先使用 rollout 里保存的 `prompt_token_ids` 和 `completion_token_ids`。这可以避免把生成文本重新 tokenize 后和原始采样 token 不一致。旧产物缺少 token ids 时才会退回文本 tokenize。

远端 `main` 不再把 `outputs/` 实验产物当代码跟踪；`outputs/` 应作为本地实验结果目录保留。

## 实验目标

first-error ablation 关注的问题是：如果 student 已经生成了一段推理，其中前缀还正确，但后面出现第一个错误，那么 teacher 只拿到局部 first-error diagnostic，是否能像拿到完整 reference solution 一样帮助模型从学生自己的前缀继续修复？

它同时看两类结果：

- continuation 指标：在学生正确前缀后继续生成，比较 `pass_at_n`、`avg_at_n`、`majority_vote`、prefix preservation、copy rate、locality 等。
- segmented KL 指标：沿 student 原始 completion token 轨迹，对比 teacher context 和 student base context 的 token 分布差异，并单独汇总 `valid_prefix` 与 `first_error_neighborhood` 两段。

## 数据流

主入口：

```bash
scripts/run_first_error_ablation.sh
```

阶段：

1. Phase 0: `eval/prepare_sample_manifest.py`
   - 输出 `$OUT/sample_indices.json`。

2. Phase A: `eval/quick_rollout_openthoughts.py`
   - 只跑 `student` condition。
   - 默认用每题 `sample_index=0` 作为 first-error judge 的学生轨迹。
   - 输出 `$OUT/student_rollouts.jsonl` 和 `$OUT/student_rollout_summary.json`。

3. Phase B: `generate_1st-error_json.py`
   - 读取 sample manifest、dataset reference solution、student rollouts。
   - 调用 DeepSeek/OpenAI-compatible chat completion 生成 first-error diagnostic。
   - 输出 `$OUT/first_error.jsonl`。

4. Phase C: `eval/quick_first_error_ablation.py --mode continuation`
   - 从 student sample-0 rollout 中定位 `prefix_valid_until` 和 `first_error_sentence`。
   - 把 `student_prefix` 作为 assistant prefill。
   - 在 `teacher_base_w_text` 和 `teacher_base_w_first_error` 两个 condition 下续写。
   - 输出 `$OUT/first_error_continuations.jsonl` 和 `$OUT/first_error_continuation_summary.json`。

5. Phase D: `eval/quick_first_error_ablation.py --mode kl`
   - target trajectory 优先使用 student rollout 的 `completion_token_ids`。
   - student base context 是原始 non-thinking problem prompt。
   - teacher context 分别是 final answer + 完整 reference solution，以及 final answer + first-error diagnostic JSON。
   - 输出 `$OUT/first_error_kl.jsonl` 和 `$OUT/first_error_kl_summary.json`。

## First-Error Diagnostic Schema

`first_error.jsonl` 每行是一个 problem 的诊断：

```json
{
  "problem_id": 123,
  "diagnostic": {
    "prefix_valid_until": "学生 trace 中最后一句仍然正确的原句",
    "first_error_sentence": "学生 trace 中第一句错误原句，若无错误则为 null",
    "error_type": "algebraic_error",
    "valid_prefix_summary": "正确前缀摘要",
    "student_plan": "学生当前计划",
    "local_repair": "不超过 80 词的局部修复",
    "next_subgoal_after_repair": "不超过 40 词的下一步局部目标"
  }
}
```

注意：

- `prefix_valid_until` 和 `first_error_sentence` 必须尽量复制 student trace 里的原句。
- 旧的 `first_error_span` schema 会被拒绝；需要用新脚本重新生成。
- 如果 `first_error_sentence` 无法在 student completion 中定位，后续切片和 KL 会报错，通常需要重新生成或手动修正该条 diagnostic。

## 正式实验怎么跑

先准备 API key。脚本默认读 `DEEPSEEK_API_KEY`，API base 默认是 `https://api.deepseek.com`：

```bash
cd /home/ruizzhao/OPSD-main
export DEEPSEEK_API_KEY="你的_API_KEY"
# 如果使用自定义 OpenAI-compatible endpoint，再设置：
# export DEEPSEEK_API_BASE="https://你的-endpoint/v1"
```

快速 smoke run：

```bash
MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/first_error_smoke_$(date +%Y%m%d_%H%M%S) \
FIRST_ERROR_MODEL=DeepSeek-v4-pro \
bash scripts/run_first_error_ablation.sh smoke \
  --gpus "4" \
  --sample-size 8 \
  --case-size 8 \
  --val-n 1 \
  --max-new-tokens 1024 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.75 \
  --hf-device-map cuda \
  --seed 0
```

正式 quick run 示例：

```bash
MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_first_error_ablation_$(date +%Y%m%d_%H%M%S) \
FIRST_ERROR_MODEL=DeepSeek-v4-pro \
bash scripts/run_first_error_ablation.sh quick \
  --gpus "4 5 6 7" \
  --sample-size 128 \
  --case-size 0 \
  --val-n 4 \
  --max-new-tokens 16384 \
  --max-model-len 20000 \
  --first-error-max-tokens 8192 \
  --gpu-memory-utilization 0.75 \
  --hf-device-map cuda \
  --seed 0
```

参数要点：

- `--gpus "4 5 6 7"`: 指定物理 GPU 列表；脚本会按 GPU 数自动设置 shard 数。
- `--sample-size`: 抽多少道题进入 student rollout 和 first-error judging。
- `--case-size 0`: 不额外 subsample first-error cases；大于 0 时只抽指定数量 cases。
- `--val-n`: continuation 每个 condition 每题采样条数。
- `--max-new-tokens`: student rollout 和 continuation 的最大生成长度。
- `--max-model-len`: vLLM rollout 和 HF KL probe 的上下文上限。
- `--first-error-model`: first-error judge 模型名，可通过参数或环境变量 `FIRST_ERROR_MODEL` 设置。
- `--hf-device-map cuda`: 每个 KL 子进程只看到当前 `CUDA_VISIBLE_DEVICES=$gpu`，因此代码里的 `cuda:0` 对应分配到的物理 GPU。

## 复用已有产物

如果已经有 sample manifest、student rollout 或 first-error diagnostic，可以跳过对应阶段：

```bash
MODEL=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
OUT=/home/ruizzhao/OPSD-main/outputs/opsd_quick/qwen31b_first_error_reuse_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_first_error_ablation.sh quick \
  --gpus "4 5 6 7" \
  --sample-indices-file /path/to/sample_indices.json \
  --student-rollout-file /path/to/student_rollouts.jsonl \
  --first-error-file /path/to/first_error.jsonl \
  --case-size 0 \
  --val-n 4 \
  --max-new-tokens 16384 \
  --max-model-len 20000 \
  --gpu-memory-utilization 0.75 \
  --hf-device-map cuda \
  --seed 0
```

复用时要确保三个文件来自同一批 problem ids，否则 `generate_1st-error_json.py` 或 first-error case selection 会因为缺失 problem 或定位不到原句而失败。

## 单独运行各阶段

生成 first-error diagnostic：

```bash
python generate_1st-error_json.py \
  --sample-indices /path/to/sample_indices.json \
  --rollout-file /path/to/student_rollouts.jsonl \
  --output-file /path/to/first_error.jsonl \
  --model DeepSeek-v4-pro \
  --max-tokens 8192
```

只跑 continuation：

```bash
CUDA_VISIBLE_DEVICES=4 python eval/quick_first_error_ablation.py \
  --mode continuation \
  --model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --student-rollout-file /path/to/student_rollouts.jsonl \
  --first-error-file /path/to/first_error.jsonl \
  --case-size 16 \
  --val-n 4 \
  --max-new-tokens 4096 \
  --max-model-len 20000 \
  --output-file /path/to/first_error_continuations.jsonl \
  --summary-file /path/to/first_error_continuation_summary.json
```

只跑 segmented KL：

```bash
CUDA_VISIBLE_DEVICES=4 python eval/quick_first_error_ablation.py \
  --mode kl \
  --model /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
  --student-rollout-file /path/to/student_rollouts.jsonl \
  --first-error-file /path/to/first_error.jsonl \
  --case-size 16 \
  --top-k 20 \
  --top-kl-positions 20 \
  --first-window-tokens 32 \
  --neighborhood-before-tokens 32 \
  --neighborhood-after-tokens 64 \
  --max-model-len 20000 \
  --hf-device-map cuda \
  --output-file /path/to/first_error_kl.jsonl \
  --summary-file /path/to/first_error_kl_summary.json
```

## 输出怎么看

`first_error_continuation_summary.json` 的主要字段在 `conditions` 下：

- `avg_at_n`: 所有 generation 的平均正确率。
- `pass_at_n`: 每题 N 条采样里至少一条正确的比例。
- `majority_vote`: 多数投票答案正确率。
- `format_rate`: 是否成功抽取 boxed answer。
- `prefix_preservation_rate`: 续写是否保留 student prefix。
- `avg_reference_copy_rate`: 和 reference solution 的复制重合程度。
- `avg_locality_score`: 局部修复倾向的粗略分数。

`first_error_kl_summary.json` 的主要字段：

- `contrasts.teacher_base_w_text_vs_student_base`: 完整 reference solution context 相对 student base 的 KL。
- `contrasts.teacher_base_w_first_error_vs_student_base`: first-error diagnostic context 相对 student base 的 KL。
- `segment_kl.*.valid_prefix`: 正确前缀 token 段的 KL 汇总。
- `segment_kl.*.first_error_neighborhood`: 第一处错误附近 token 段的 KL 汇总。

如果 first-error diagnostic 有效但 KL 特别集中在 `first_error_neighborhood`，说明 teacher 信息主要改变了错误附近的 token 分布；如果 `valid_prefix` 也明显变化，可能说明 intervention 太强，影响了学生原本正确前缀。

## 本地验证

只验证这次新增逻辑的单测：

```bash
python -m unittest tests/test_first_error_ablation.py
```

查看 CLI 参数：

```bash
python eval/quick_first_error_ablation.py --help
```

## 常见问题

- `Missing API key: set DEEPSEEK_API_KEY`: 需要先导出 `DEEPSEEK_API_KEY`，或用 `--api-key-env` 指向别的环境变量。
- `first_error_sentence was not found in the student completion`: diagnostic 里的句子没有精确或规范化匹配 student completion，建议重跑或修正该条 `first_error.jsonl`。
- `regenerate first-error diagnostics with the sentence schema`: 说明用的是旧 schema，重新跑 `generate_1st-error_json.py`。
- vLLM 启动显存不足：把 `--gpu-memory-utilization` 降到 `0.75` 或 `0.8`，或减少并发 GPU 数。
- `outputs/` 不应提交到 Git。正式结果放本地 `outputs/opsd_quick/...`，代码仓库只提交脚本、测试和文档。
