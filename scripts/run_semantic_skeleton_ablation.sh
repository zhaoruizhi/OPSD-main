#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
if [[ $# -gt 0 ]]; then
  shift
fi

MODEL="${MODEL:-/data0/shared/Qwen3-1.7B}"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
SPLIT="${SPLIT:-train}"
OUT="${OUT:-/data1/opsd_quick/qwen31b_skeleton_ablation_$(date +%Y%m%d_%H%M%S)}"
SAMPLE_INDICES_FILE=""
SKELETON_FILE=""
SAMPLE_SIZE=128
LOGIT_SIZE=0
VAL_N=4
MAX_NEW_TOKENS=1024
TEMPERATURE=1.1
TOP_P=0.95
TOP_K=20
SEED=0
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=20000
PROBE_TOKENS=0
TRAJECTORY_SAMPLE_INDEX=0
SKELETON_MAX_TOKENS=2048
SKELETON_BACKEND="${SKELETON_BACKEND:-api}"
SKELETON_MODEL="${SKELETON_MODEL:-}"
SKELETON_GPUS="${SKELETON_GPUS:-}"
SKELETON_API_CONCURRENCY="${SKELETON_API_CONCURRENCY:-8}"
SKELETON_TIMEOUT="${SKELETON_TIMEOUT:-300}"
SKELETON_MAX_RETRIES="${SKELETON_MAX_RETRIES:-2}"
SKELETON_FLUSH_EVERY="${SKELETON_FLUSH_EVERY:-10}"
SKELETON_RETRY_DELAY="${SKELETON_RETRY_DELAY:-2.0}"
SKELETON_MAX_RETRY_DELAY="${SKELETON_MAX_RETRY_DELAY:-60.0}"
SKELETON_RESPONSE_FORMAT_JSON="${SKELETON_RESPONSE_FORMAT_JSON:-0}"
SKELETON_API_DISABLE_THINKING="${SKELETON_API_DISABLE_THINKING:-0}"
SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES="${SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES:-50}"
SKELETON_VLLM_TENSOR_PARALLEL_SIZE="${SKELETON_VLLM_TENSOR_PARALLEL_SIZE:-1}"
SKELETON_VLLM_GPU_MEMORY_UTILIZATION="${SKELETON_VLLM_GPU_MEMORY_UTILIZATION:-$GPU_MEMORY_UTILIZATION}"
SKELETON_VLLM_MAX_MODEL_LEN="${SKELETON_VLLM_MAX_MODEL_LEN:-$MAX_MODEL_LEN}"
SKELETON_VLLM_TOP_P="${SKELETON_VLLM_TOP_P:-1.0}"
SKELETON_VLLM_TOP_K="${SKELETON_VLLM_TOP_K:--1}"
SKELETON_VLLM_ENABLE_THINKING="${SKELETON_VLLM_ENABLE_THINKING:-0}"
SKIP_ROLLOUT_ENTROPY=0
HF_DEVICE_MAP="${HF_DEVICE_MAP:-cuda}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"

case "$MODE" in
  smoke)
    SAMPLE_SIZE=8
    LOGIT_SIZE=4
    ;;
  quick)
    SAMPLE_SIZE=128
    LOGIT_SIZE=0
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
    --skeleton-file)
      SKELETON_FILE="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
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
    --trajectory-sample-index)
      TRAJECTORY_SAMPLE_INDEX="$2"
      shift 2
      ;;
    --skeleton-max-tokens)
      SKELETON_MAX_TOKENS="$2"
      shift 2
      ;;
    --skeleton-backend)
      SKELETON_BACKEND="$2"
      shift 2
      ;;
    --skeleton-model)
      SKELETON_MODEL="$2"
      shift 2
      ;;
    --skeleton-gpus)
      SKELETON_GPUS="$2"
      shift 2
      ;;
    --skeleton-timeout)
      SKELETON_TIMEOUT="$2"
      shift 2
      ;;
    --skeleton-max-retries)
      SKELETON_MAX_RETRIES="$2"
      shift 2
      ;;
    --skeleton-flush-every)
      SKELETON_FLUSH_EVERY="$2"
      shift 2
      ;;
    --skeleton-retry-delay)
      SKELETON_RETRY_DELAY="$2"
      shift 2
      ;;
    --skeleton-max-retry-delay)
      SKELETON_MAX_RETRY_DELAY="$2"
      shift 2
      ;;
    --skeleton-abort-after-consecutive-failures)
      SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES="$2"
      shift 2
      ;;
    --skeleton-response-format-json)
      SKELETON_RESPONSE_FORMAT_JSON=1
      shift
      ;;
    --skeleton-no-response-format-json)
      SKELETON_RESPONSE_FORMAT_JSON=0
      shift
      ;;
    --skeleton-api-disable-thinking)
      SKELETON_API_DISABLE_THINKING=1
      shift
      ;;
    --skeleton-api-enable-thinking)
      SKELETON_API_DISABLE_THINKING=0
      shift
      ;;
    --skeleton-vllm-tensor-parallel-size)
      SKELETON_VLLM_TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --skeleton-vllm-gpu-memory-utilization)
      SKELETON_VLLM_GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --skeleton-vllm-max-model-len)
      SKELETON_VLLM_MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --skeleton-vllm-top-p)
      SKELETON_VLLM_TOP_P="$2"
      shift 2
      ;;
    --skeleton-vllm-top-k)
      SKELETON_VLLM_TOP_K="$2"
      shift 2
      ;;
    --skeleton-enable-thinking)
      SKELETON_VLLM_ENABLE_THINKING=1
      shift
      ;;
    --skeleton-disable-thinking)
      SKELETON_VLLM_ENABLE_THINKING=0
      shift
      ;;
    --skip-rollout-entropy)
      SKIP_ROLLOUT_ENTROPY=1
      shift
      ;;
    --keep-rollout-entropy)
      SKIP_ROLLOUT_ENTROPY=0
      shift
      ;;
    --hf-device-map)
      HF_DEVICE_MAP="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
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
echo "Dataset: $DATASET:$SPLIT"
echo "Mode: $MODE"
echo "Sample size: $SAMPLE_SIZE | Val-N: $VAL_N | Logit size: $LOGIT_SIZE"
echo "Phase 3: HF device map=$HF_DEVICE_MAP | Skip rollout entropy=$SKIP_ROLLOUT_ENTROPY"
read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARRAY[@]}"
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "No GPU ids configured. Set GPU_IDS or pass --gpus." >&2
  exit 2
fi
if [[ -z "$SKELETON_GPUS" ]]; then
  SKELETON_GPUS="${GPU_ID_ARRAY[0]}"
fi
SKELETON_MODEL_FOR_RUN="$SKELETON_MODEL"
if [[ -z "$SKELETON_MODEL_FOR_RUN" ]]; then
  if [[ "$SKELETON_BACKEND" == "vllm" ]]; then
    SKELETON_MODEL_FOR_RUN="$MODEL"
  else
    SKELETON_MODEL_FOR_RUN="deepseek-v4-pro"
  fi
fi
echo "GPU ids: ${GPU_ID_ARRAY[*]} | Num shards: $NUM_SHARDS"
echo "Skeleton backend: $SKELETON_BACKEND | Skeleton model: $SKELETON_MODEL_FOR_RUN"
if [[ "$SKELETON_BACKEND" == "vllm" ]]; then
  echo "Skeleton vLLM GPUs: $SKELETON_GPUS | TP: $SKELETON_VLLM_TENSOR_PARALLEL_SIZE"
else
  echo "Skeleton API concurrency: $SKELETON_API_CONCURRENCY | Timeout: $SKELETON_TIMEOUT | Max retries: $SKELETON_MAX_RETRIES"
  echo "Skeleton JSON response_format: $SKELETON_RESPONSE_FORMAT_JSON | API disable thinking: $SKELETON_API_DISABLE_THINKING | Abort after failures: $SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES"
fi

echo
echo "== Phase 0: fixed 128-sample manifest =="
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
echo "== Phase 1: semantic skeleton generation =="
if [[ -n "$SKELETON_FILE" ]]; then
  cp "$SKELETON_FILE" "$OUT/skeletons.jsonl"
else
  SKELETON_GENERATE_ARGS=(
    eval/generate_semantic_skeletons.py
    --dataset "$DATASET"
    --split "$SPLIT"
    --sample-indices-file "$OUT/sample_indices.json"
    --output-file "$OUT/skeletons.jsonl"
    --skeleton-backend "$SKELETON_BACKEND"
    --skeleton-model "$SKELETON_MODEL_FOR_RUN"
    --max-tokens "$SKELETON_MAX_TOKENS"
    --timeout "$SKELETON_TIMEOUT"
    --max-retries "$SKELETON_MAX_RETRIES"
    --flush-every "$SKELETON_FLUSH_EVERY"
    --retry-delay "$SKELETON_RETRY_DELAY"
    --max-retry-delay "$SKELETON_MAX_RETRY_DELAY"
    --abort-after-consecutive-failures "$SKELETON_ABORT_AFTER_CONSECUTIVE_FAILURES"
  )
  if [[ "$SKELETON_BACKEND" == "vllm" ]]; then
    SKELETON_GENERATE_ARGS+=(
      --vllm-tensor-parallel-size "$SKELETON_VLLM_TENSOR_PARALLEL_SIZE"
      --vllm-gpu-memory-utilization "$SKELETON_VLLM_GPU_MEMORY_UTILIZATION"
      --vllm-max-model-len "$SKELETON_VLLM_MAX_MODEL_LEN"
      --vllm-top-p "$SKELETON_VLLM_TOP_P"
      --vllm-top-k "$SKELETON_VLLM_TOP_K"
    )
    if [[ "$SKELETON_VLLM_ENABLE_THINKING" == "1" ]]; then
      SKELETON_GENERATE_ARGS+=(--vllm-enable-thinking)
    fi
    CUDA_VISIBLE_DEVICES="$SKELETON_GPUS" python "${SKELETON_GENERATE_ARGS[@]}"
  else
    SKELETON_GENERATE_ARGS+=(--api-concurrency "$SKELETON_API_CONCURRENCY")
    if [[ "$SKELETON_RESPONSE_FORMAT_JSON" == "1" ]]; then
      SKELETON_GENERATE_ARGS+=(--response-format-json)
    fi
    if [[ "$SKELETON_API_DISABLE_THINKING" == "1" ]]; then
      SKELETON_GENERATE_ARGS+=(--api-disable-thinking)
    fi
    python "${SKELETON_GENERATE_ARGS[@]}"
  fi
fi

echo
echo "== Phase 2: standalone rollouts =="
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
    --skeleton-file "$OUT/skeletons.jsonl" \
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
echo "== Phase 3: full-response logit distribution probe =="
LOGIT_EXTRA_ARGS=(--hf-device-map "$HF_DEVICE_MAP")
if [[ "$SKIP_ROLLOUT_ENTROPY" == "1" ]]; then
  LOGIT_EXTRA_ARGS+=(--skip-rollout-entropy)
fi
pids=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_logit_probe.py \
    --model "$MODEL" \
    --rollout-file "$OUT/rollouts.jsonl" \
    --skeleton-file "$OUT/skeletons.jsonl" \
    --trajectory-condition teacher_base \
    --trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX" \
    --logit-size "$LOGIT_SIZE" \
    --probe-tokens "$PROBE_TOKENS" \
    --seed "$SEED" \
    --top-k "$TOP_K" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    "${LOGIT_EXTRA_ARGS[@]}" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --output-file "$OUT/logit_probe_shard${gpu}.jsonl" \
    --summary-file "$OUT/logit_summary_shard${gpu}.json" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
cat "$OUT"/logit_probe_shard*.jsonl > "$OUT/logit_probe.jsonl"
python eval/quick_logit_probe.py \
  --summarize-only \
  --input-file "$OUT/logit_probe.jsonl" \
  --summary-file "$OUT/logit_summary.json"

echo
echo "Semantic skeleton ablation complete."
echo "Sample manifest:  $OUT/sample_indices.json"
echo "Skeletons:        $OUT/skeletons.jsonl"
echo "Rollout summary:  $OUT/rollout_summary.json"
echo "Logit summary:    $OUT/logit_summary.json"
