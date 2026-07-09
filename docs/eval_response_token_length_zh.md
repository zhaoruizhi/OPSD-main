# 统计评估 response 的平均 token length

本文档说明如何统计 `eval/evaluate_math.py` 生成的评估结果 JSON 中，模型 response 的平均 token length，用来对比 reference 和 skeleton 在 `avg@12` 评估下的输出长度。

## 统计口径

评估结果文件通常保存在：

```text
eval_results/aime25_reference_checkpoint_25.json
eval_results/aime25_reference_checkpoint_50.json
eval_results/aime25_skeleton_checkpoint_25.json
eval_results/aime25_skeleton_checkpoint_50.json
...
```

当前 `eval/evaluate_math.py` 的输出结构是：

```json
{
  "dataset": "aime25",
  "val_n": 12,
  "average_at_n_pct": 43.1,
  "results": [
    {
      "problem_id": 1,
      "generations": [
        {
          "predicted_answer": "103",
          "full_generation": "<think>...",
          "correct": true,
          "formatted": true
        }
      ],
      "full_generation": "<think>..."
    }
  ]
}
```

统计脚本 `scripts/count_eval_response_tokens.py` 的默认口径是：

- 统计评估阶段模型生成的完整 response，也就是 `results[*].generations[*].full_generation`。
- 对 `val_n=12` 的文件，会统计每道题的 12 条 generation，因此 AIME/HMMT 30 题通常是 `30 * 12 = 360` 条 response。
- 不统计 `problem`、prompt、ground truth、`predicted_answer` 或其他指标字段。
- 使用传入的 HuggingFace tokenizer 统计 token 数。
- tokenize 时使用 `add_special_tokens=False`，避免额外加 BOS/EOS 等包装 token。
- 一个评估文件的平均长度为：

```text
average_response_tokens = sum(token_count(full_generation)) / response_count
```

脚本也兼容旧格式：如果某道题没有 `generations` 列表，但有顶层 `results[*].full_generation`，会退回统计这个字段。

## 代码具体做了什么

核心逻辑在 `scripts/count_eval_response_tokens.py`：

- `discover_eval_files(...)`
  - 接收一个或多个评估 JSON 文件或目录。
  - 如果传入目录，只读取文件名形如 `<dataset>_<reference|skeleton>_checkpoint_<step>.json` 的 JSON。
  - 支持用 `--datasets`、`--conditions`、`--target-steps` 过滤。

- `summarize_eval_file(path, tokenizer)`
  - 读取单个评估 JSON。
  - 从文件名解析 `dataset`、`condition`、`checkpoint_step`。
  - 提取所有 response 文本。
  - 计算 `response_count`、`total_response_tokens`、`average_response_tokens`。
  - 同时保留 `average_at_n_pct`、`pass_at_n_pct`、`majority_vote_at_n_pct`、`format_rate`，方便和 avg@12 表格交叉检查。

- `extract_response_texts(payload, path)`
  - 优先读取 `results[*].generations[*].full_generation`。
  - 若旧产物没有 nested `generations`，则读取 `results[*].full_generation`。
  - 如果缺少 response 字段，会直接报错，避免静默统计错字段。

- `token_count(tokenizer, text)`
  - 优先调用 `tokenizer.encode(text, add_special_tokens=False)`。
  - 如果 tokenizer 没有 `encode`，则退回 `tokenizer(text, add_special_tokens=False)["input_ids"]`。

## 服务器运行步骤

先进入项目目录并激活训练或评估环境：

```bash
cd /home/ruizzhao/OPSD-main
conda activate opsd
```

确认 `transformers` 可用：

```bash
python -c "import transformers; print(transformers.__version__)"
```

准备 tokenizer 和评估结果目录。`TOKENIZER` 建议使用评估时的 base model 路径，避免联网下载：

```bash
TOKENIZER=/home/ruizzhao/OPSD-main/models/Qwen3-1.7B
EVAL_DIR=eval/eval_results
```

如果你的评估结果目录就在项目根目录下，也可以改成：

```bash
EVAL_DIR=eval_results
```

统计 AIME25 的 reference 和 skeleton，并输出 CSV/JSON：

```bash
python scripts/count_eval_response_tokens.py "$EVAL_DIR" \
  --tokenizer "$TOKENIZER" \
  --datasets aime25 \
  --conditions reference skeleton \
  --target-steps 25 50 75 100 \
  --precision 1 \
  --trust-remote-code \
  --output-csv eval_response_token_lengths_aime25.csv \
  --output-json eval_response_token_lengths_aime25.json
```

终端会输出类似下面的表格：

```text
Average generated response token length
Dataset: aime25
| condition | 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| reference | 12345.6 | 12001.2 | 11888.4 | 11920.7 |
| skeleton | 11011.3 | 10950.9 | 11100.0 | 10880.5 |
```

这些值就是可以填到表格里 `Token length` 列的平均 response token length。

如果要一次统计 AIME24、AIME25、HMMT25：

```bash
python scripts/count_eval_response_tokens.py "$EVAL_DIR" \
  --tokenizer "$TOKENIZER" \
  --datasets aime24 aime25 hmmt25 \
  --conditions reference skeleton \
  --target-steps 25 50 75 100 \
  --precision 1 \
  --trust-remote-code \
  --output-csv eval_response_token_lengths_all.csv \
  --output-json eval_response_token_lengths_all.json
```

如果你是在 `eval/` 目录下运行命令：

```bash
cd /home/ruizzhao/OPSD-main/eval

python ../scripts/count_eval_response_tokens.py eval_results \
  --tokenizer "$TOKENIZER" \
  --datasets aime25 \
  --conditions reference skeleton \
  --target-steps 25 50 75 100 \
  --precision 1 \
  --trust-remote-code
```

## 输出文件说明

CSV/JSON 每一行对应一个数据集、一个条件、一个 checkpoint：

```text
dataset,condition,checkpoint_step,eval_file,problem_count,val_n,response_count,total_response_tokens,average_response_tokens,average_at_n_pct,pass_at_n_pct,majority_vote_at_n_pct,format_rate
aime25,reference,50,eval_results/aime25_reference_checkpoint_50.json,30,12,360,4320432,12001.2,43.1,80.0,60.0,99.4
```

字段含义：

- `dataset`: 数据集名，例如 `aime25`。
- `condition`: `reference` 或 `skeleton`。
- `checkpoint_step`: checkpoint step，例如 `25/50/75/100`。
- `eval_file`: 被统计的评估 JSON 文件路径。
- `problem_count`: 题目数。
- `val_n`: 每题采样数，当前评估通常是 `12`。
- `response_count`: 实际统计的 response 数，通常是 `problem_count * val_n`。
- `total_response_tokens`: 所有 response 的 token 总数。
- `average_response_tokens`: 平均 response token length。
- `average_at_n_pct`: 评估 JSON 里保存的 Avg@N 指标，用于和原表里的 `avg@12` 对齐检查。
- `pass_at_n_pct`、`majority_vote_at_n_pct`、`format_rate`: 评估 JSON 里的其他指标，方便排查。

## 常见检查

确认评估文件名符合脚本默认解析规则：

```bash
ls "$EVAL_DIR" | grep 'aime25_.*checkpoint'
```

如果脚本报 `No matching evaluation JSON files found.`，通常是 `EVAL_DIR` 不对，或者文件名不是 `<dataset>_<reference|skeleton>_checkpoint_<step>.json`。

如果脚本报 `missing string field 'full_generation'`，说明评估 JSON 不是当前 `evaluate_math.py` 保存的格式，或者某条 generation 缺少完整 response，需要先打开对应 JSON 检查。

如果 tokenizer 下载失败，优先把 `--tokenizer` 指向服务器本地 base model 路径，例如 JSON 顶部 `base_model` 字段对应的目录。
