#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
if [[ $# -gt 0 ]]; then
  shift
fi

MODEL="${MODEL:-/data0/shared/Qwen3-1.7B}"
OUT="${OUT:-/data1/opsd_quick/qwen31b_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_SIZE=""
PREFIX_SIZE=""
LOGIT_SIZE=""
VAL_N=4
MAX_NEW_TOKENS=1024
TEMPERATURE=1.1
TOP_P=0.95
TOP_K=20
SEED=0
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=20000
PROBE_TOKENS=0

case "$MODE" in
  smoke)
    SAMPLE_SIZE=32
    PREFIX_SIZE=16
    LOGIT_SIZE=16
    ;;
  quick)
    SAMPLE_SIZE=256
    PREFIX_SIZE=64
    LOGIT_SIZE=64
    ;;
  *)
    echo "Usage: $0 {smoke|quick} [--model PATH] [--out DIR] [options]" >&2
    exit 2
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --prefix-size)
      PREFIX_SIZE="$2"
      shift 2
      ;;
    --logit-size)
      LOGIT_SIZE="$2"
      shift 2
      ;;
    --val-n)
      VAL_N="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --probe-tokens)
      PROBE_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"

echo "Output directory: $OUT"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "Sample size: $SAMPLE_SIZE | Prefix size: $PREFIX_SIZE | Logit size: $LOGIT_SIZE"

echo
echo "== Phase A: standalone rollouts =="
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_rollout_openthoughts.py \
    --model "$MODEL" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --shard-id "$gpu" \
    --num-shards 4 \
    --val-n "$VAL_N" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output-file "$OUT/rollout_shard${gpu}.jsonl" \
    --summary-file "$OUT/rollout_summary_shard${gpu}.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "$OUT"/rollout_shard*.jsonl > "$OUT/rollouts.jsonl"
python eval/quick_rollout_openthoughts.py \
  --summarize-only \
  --input-file "$OUT/rollouts.jsonl" \
  --summary-file "$OUT/rollout_summary.json"

echo
echo "== Phase B: prefix-conditioned continuations =="
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_prefix_intervention.py \
    --model "$MODEL" \
    --student-rollout-file "$OUT/rollouts.jsonl" \
    --prefix-size "$PREFIX_SIZE" \
    --seed "$SEED" \
    --shard-id "$gpu" \
    --num-shards 4 \
    --val-n 1 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output-file "$OUT/prefix_shard${gpu}.jsonl" \
    --summary-file "$OUT/prefix_summary_shard${gpu}.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "$OUT"/prefix_shard*.jsonl > "$OUT/prefix_cases.jsonl"
python eval/quick_prefix_intervention.py \
  --summarize-only \
  --input-file "$OUT/prefix_cases.jsonl" \
  --summary-file "$OUT/prefix_summary.json"

echo
echo "== Phase C: full-response logit distribution probe =="
CUDA_VISIBLE_DEVICES=0 python eval/quick_logit_probe.py \
  --model "$MODEL" \
  --rollout-file "$OUT/rollouts.jsonl" \
  --logit-size "$LOGIT_SIZE" \
  --probe-tokens "$PROBE_TOKENS" \
  --seed "$SEED" \
  --top-k "$TOP_K" \
  --max-context-tokens "$MAX_MODEL_LEN" \
  --output-file "$OUT/logit_probe.jsonl" \
  --summary-file "$OUT/logit_summary.json"

echo
echo "Quick OPSD probe complete."
echo "Rollout summary: $OUT/rollout_summary.json"
echo "Prefix summary:  $OUT/prefix_summary.json"
echo "Logit summary:   $OUT/logit_summary.json"
