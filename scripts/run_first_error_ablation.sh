#!/usr/bin/env bash
set -euo pipefail

MODE="quick"
if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi

MODEL="${MODEL:-/data0/shared/Qwen3-1.7B}"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
SPLIT="${SPLIT:-train}"
OUT="${OUT:-/data1/opsd_quick/qwen31b_first_error_ablation_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_INDICES_FILE=""
STUDENT_ROLLOUT_FILE=""
FIRST_ERROR_FILE=""
SAMPLE_SIZE=128
CASE_SIZE=0
VAL_N=4
MAX_NEW_TOKENS=1024
TEMPERATURE=1.1
TOP_P=0.95
TOP_K=20
SEED=0
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=20000
HF_DEVICE_MAP="${HF_DEVICE_MAP:-cuda}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"
FIRST_ERROR_MODEL="${FIRST_ERROR_MODEL:-DeepSeek-v4-pro}"
FIRST_ERROR_MAX_TOKENS=8192
TOP_KL_POSITIONS=20
FIRST_WINDOW_TOKENS=32
NEIGHBORHOOD_BEFORE_TOKENS=32
NEIGHBORHOOD_AFTER_TOKENS=64

case "$MODE" in
  smoke)
    SAMPLE_SIZE=8
    ;;
  quick)
    SAMPLE_SIZE=128
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
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --sample-indices-file)
      SAMPLE_INDICES_FILE="$2"
      shift 2
      ;;
    --student-rollout-file)
      STUDENT_ROLLOUT_FILE="$2"
      shift 2
      ;;
    --first-error-file)
      FIRST_ERROR_FILE="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --case-size)
      CASE_SIZE="$2"
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
    --hf-device-map)
      HF_DEVICE_MAP="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --first-error-model)
      FIRST_ERROR_MODEL="$2"
      shift 2
      ;;
    --first-error-max-tokens)
      FIRST_ERROR_MAX_TOKENS="$2"
      shift 2
      ;;
    --top-kl-positions)
      TOP_KL_POSITIONS="$2"
      shift 2
      ;;
    --first-window-tokens)
      FIRST_WINDOW_TOKENS="$2"
      shift 2
      ;;
    --neighborhood-before-tokens)
      NEIGHBORHOOD_BEFORE_TOKENS="$2"
      shift 2
      ;;
    --neighborhood-after-tokens)
      NEIGHBORHOOD_AFTER_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT"
read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARRAY[@]}"
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "No GPU ids configured. Set GPU_IDS or pass --gpus." >&2
  exit 2
fi

echo "Output directory: $OUT"
echo "Model: $MODEL"
echo "Dataset: $DATASET:$SPLIT"
echo "Mode: $MODE"
echo "Sample size: $SAMPLE_SIZE | Case size: $CASE_SIZE | Val-N: $VAL_N"
echo "GPU ids: ${GPU_ID_ARRAY[*]} | Num shards: $NUM_SHARDS"

echo
echo "== Phase 0: sample manifest =="
if [[ -n "$SAMPLE_INDICES_FILE" ]]; then
  cp "$SAMPLE_INDICES_FILE" "$OUT/sample_indices.json"
else
  python eval/prepare_sample_manifest.py \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --sample-size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --output-file "$OUT/sample_indices.json"
fi

echo
echo "== Phase A: student_base rollouts =="
if [[ -n "$STUDENT_ROLLOUT_FILE" ]]; then
  cp "$STUDENT_ROLLOUT_FILE" "$OUT/student_rollouts.jsonl"
  python eval/quick_rollout_openthoughts.py \
    --summarize-only \
    --input-file "$OUT/student_rollouts.jsonl" \
    --summary-file "$OUT/student_rollout_summary.json"
else
  pids=()
  for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
    gpu="${GPU_ID_ARRAY[$gpu_index]}"
    shard_id="$gpu_index"
    CUDA_VISIBLE_DEVICES=$gpu python eval/quick_rollout_openthoughts.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --sample-size "$SAMPLE_SIZE" \
      --sample-indices-file "$OUT/sample_indices.json" \
      --seed "$SEED" \
      --shard-id "$shard_id" \
      --num-shards "$NUM_SHARDS" \
      --val-n "$VAL_N" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --condition student \
      --output-file "$OUT/student_rollout_shard${gpu}.jsonl" \
      --summary-file "$OUT/student_rollout_summary_shard${gpu}.json" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  cat "$OUT"/student_rollout_shard*.jsonl > "$OUT/student_rollouts.jsonl"
  python eval/quick_rollout_openthoughts.py \
    --summarize-only \
    --input-file "$OUT/student_rollouts.jsonl" \
    --summary-file "$OUT/student_rollout_summary.json"
fi

echo
echo "== Phase B: first-error diagnostics =="
if [[ -n "$FIRST_ERROR_FILE" ]]; then
  cp "$FIRST_ERROR_FILE" "$OUT/first_error.jsonl"
else
  python generate_1st-error_json.py \
    --sample-indices "$OUT/sample_indices.json" \
    --rollout-file "$OUT/student_rollouts.jsonl" \
    --output-file "$OUT/first_error.jsonl" \
    --model "$FIRST_ERROR_MODEL" \
    --max-tokens "$FIRST_ERROR_MAX_TOKENS"
fi

echo
echo "== Phase C: first-error continuations =="
pids=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_first_error_ablation.py \
    --mode continuation \
    --model "$MODEL" \
    --student-rollout-file "$OUT/student_rollouts.jsonl" \
    --first-error-file "$OUT/first_error.jsonl" \
    --case-size "$CASE_SIZE" \
    --seed "$SEED" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --val-n "$VAL_N" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --neighborhood-before-tokens "$NEIGHBORHOOD_BEFORE_TOKENS" \
    --neighborhood-after-tokens "$NEIGHBORHOOD_AFTER_TOKENS" \
    --output-file "$OUT/first_error_continuation_shard${gpu}.jsonl" \
    --summary-file "$OUT/first_error_continuation_summary_shard${gpu}.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "$OUT"/first_error_continuation_shard*.jsonl > "$OUT/first_error_continuations.jsonl"
python eval/quick_first_error_ablation.py \
  --summarize-only \
  --summary-kind generation \
  --input-file "$OUT/first_error_continuations.jsonl" \
  --summary-file "$OUT/first_error_continuation_summary.json"

echo
echo "== Phase D: first-error segmented KL =="
pids=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_first_error_ablation.py \
    --mode kl \
    --model "$MODEL" \
    --student-rollout-file "$OUT/student_rollouts.jsonl" \
    --first-error-file "$OUT/first_error.jsonl" \
    --case-size "$CASE_SIZE" \
    --seed "$SEED" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --top-k "$TOP_K" \
    --hf-device-map "$HF_DEVICE_MAP" \
    --top-kl-positions "$TOP_KL_POSITIONS" \
    --first-window-tokens "$FIRST_WINDOW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --neighborhood-before-tokens "$NEIGHBORHOOD_BEFORE_TOKENS" \
    --neighborhood-after-tokens "$NEIGHBORHOOD_AFTER_TOKENS" \
    --output-file "$OUT/first_error_kl_shard${gpu}.jsonl" \
    --summary-file "$OUT/first_error_kl_summary_shard${gpu}.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "$OUT"/first_error_kl_shard*.jsonl > "$OUT/first_error_kl.jsonl"
python eval/quick_first_error_ablation.py \
  --summarize-only \
  --summary-kind kl \
  --input-file "$OUT/first_error_kl.jsonl" \
  --summary-file "$OUT/first_error_kl_summary.json"

echo
echo "First-error ablation complete."
echo "Student summary:       $OUT/student_rollout_summary.json"
echo "First-error JSON:      $OUT/first_error.jsonl"
echo "Continuation summary:  $OUT/first_error_continuation_summary.json"
echo "Segmented KL summary:  $OUT/first_error_kl_summary.json"
