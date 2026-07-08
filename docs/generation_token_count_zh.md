# 统计训练 generations 的累计 generated token

本文档说明如何统计 OPSD 训练输出目录中 `generations/generations_step_*.json` 的累计 generated token 数，用来填写 25/50/75/100 step 的 token efficiency 表格。

## 统计口径

训练时 `OPSDTrainer._save_generation_outputs()` 会在输出目录下保存：

```text
<EXP_DIR>/generations/generations_step_5.json
<EXP_DIR>/generations/generations_step_10.json
...
```

每个 JSON 文件的结构是：

```json
{
  "step": 25,
  "num_samples": 48,
  "generations": [
    {
      "step": 24,
      "prompt": "...",
      "completion": "..."
    }
  ]
}
```

统计脚本 `scripts/count_generation_tokens.py` 的默认口径是：

- 只统计模型生成输出，也就是每条记录的 `completion` 字段。
- 不统计 `prompt`。
- 使用 `transformers.AutoTokenizer` 加载你传入的 tokenizer。
- tokenize 时使用 `add_special_tokens=False`，避免额外加 BOS/EOS 等包装 token。
- 对每个 `generations_step_<N>.json` 先求该文件内所有 `completion` 的 token 总数。
- 对目标 step 做累计：
  - 25 step = `generations_step_5/10/15/20/25.json` 的 token 总和。
  - 50 step = 所有 `step <= 50` 的 generation 文件 token 总和。
  - 75、100 同理。

如果某个目标 step 没有对应的 `generations_step_<step>.json`，脚本仍会统计 `step <= target_step` 的已有文件，并在终端打印 warning。

## 服务器运行步骤

先进入项目目录并激活训练环境：

```bash
cd /path/to/OPSD-main
conda activate opsd
```

确认 `transformers` 可用：

```bash
python -c "import transformers; print(transformers.__version__)"
```

准备路径。`TOKENIZER` 建议使用训练时的 base model tokenizer，例如 Qwen3-1.7B：

```bash
TOKENIZER=/data0/shared/Qwen3-1.7B
REFERENCE_EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005
SKELETON_EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005
```

同时统计 reference 和 skeleton，并输出原始 token 数：

```bash
python scripts/count_generation_tokens.py \
  reference="$REFERENCE_EXP_DIR" \
  skeleton="$SKELETON_EXP_DIR" \
  --tokenizer "$TOKENIZER" \
  --target-steps 25 50 75 100 \
  --output-csv token_efficiency_counts.csv \
  --output-json token_efficiency_counts.json
```

终端会输出类似下面的 Markdown 表格：

```text
Cumulative generated completion tokens (raw tokens)
| run | 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| reference | 123456 | 234567 | 345678 | 456789 |
| skeleton | 120000 | 230000 | 340000 | 450000 |
```

如果论文图或表格需要以 `10^6` 为单位，可以加 `--scale million`：

```bash
python scripts/count_generation_tokens.py \
  reference="$REFERENCE_EXP_DIR" \
  skeleton="$SKELETON_EXP_DIR" \
  --tokenizer "$TOKENIZER" \
  --target-steps 25 50 75 100 \
  --scale million \
  --precision 3
```

这时表格里的值就是 raw token / 1,000,000。

## 只统计单个训练输出

如果只想先检查一个 run：

```bash
EXP_DIR=/data0/siyanz/opsd/qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005
TOKENIZER=/data0/shared/Qwen3-1.7B

python scripts/count_generation_tokens.py \
  skeleton="$EXP_DIR" \
  --tokenizer "$TOKENIZER" \
  --target-steps 25 50 75 100
```

也可以直接传 `generations` 目录：

```bash
python scripts/count_generation_tokens.py \
  skeleton="$EXP_DIR/generations" \
  --tokenizer "$TOKENIZER" \
  --target-steps 25 50 75 100
```

## 输出文件说明

如果传入 `--output-csv token_efficiency_counts.csv`，CSV 中每一行对应一个 run 的一个目标 step：

```text
run,generations_dir,target_step,cumulative_generated_tokens,cumulative_samples,cumulative_files,has_exact_generation_file
reference,/data0/.../generations,25,123456,240,5,True
```

字段含义：

- `run`: 命令行里 `label=PATH` 的 label，例如 `reference` 或 `skeleton`。
- `generations_dir`: 实际读取的 `generations` 目录。
- `target_step`: 目标 step，例如 25。
- `cumulative_generated_tokens`: 到该 step 为止的累计 generated completion token 数。
- `cumulative_samples`: 到该 step 为止累计统计了多少条 generation 记录。
- `cumulative_files`: 到该 step 为止累计包含了多少个 `generations_step_*.json` 文件。
- `has_exact_generation_file`: 是否存在精确的 `generations_step_<target_step>.json`。

`--output-json` 输出同样信息，方便后续画图或做 notebook 分析。

## 常见检查

确认某个 run 是否有 generation 文件：

```bash
ls "$SKELETON_EXP_DIR/generations" | head
ls "$SKELETON_EXP_DIR/generations/generations_step_25.json"
```

如果脚本报 `missing string field 'completion'`，说明某条 generation 记录不是当前训练代码保存的格式，或者文件损坏。先打开对应 JSON 检查该条记录是否包含 `completion` 字段。

如果 tokenizer 下载失败，优先使用服务器上的本地 base model 路径作为 `--tokenizer`，不要依赖联网下载。
