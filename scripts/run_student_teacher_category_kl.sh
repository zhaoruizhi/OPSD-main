#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-/data0/shared/Qwen3-1.7B}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
SPLIT="${SPLIT:-train}"
OUT="${OUT:-/data1/opsd_quick/qwen31b_student_teacher_category_kl_$(date +%Y%m%d_%H%M%S)}"
SKELETON_FILE="${SKELETON_FILE:-}"
SAMPLE_INDICES_FILE=""
SAMPLE_SIZE=10
VAL_N=1
STUDENT_TM="${STUDENT_TM:-off}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
PROBE_TOKENS="${PROBE_TOKENS:-0}"
TRAJECTORY_SAMPLE_INDEX="${TRAJECTORY_SAMPLE_INDEX:-0}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-cuda}"
GPU_IDS="${GPU_IDS:-4}"
TEACHER_CONTINUATION_TOP_N="${TEACHER_CONTINUATION_TOP_N:-10}"
TEACHER_CONTINUATION_MAX_NEW_TOKENS="${TEACHER_CONTINUATION_MAX_NEW_TOKENS:-20}"
SKIP_TEACHER_CONTINUATIONS="${SKIP_TEACHER_CONTINUATIONS:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
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
    --skeleton-file)
      SKELETON_FILE="$2"
      shift 2
      ;;
    --sample-indices-file)
      SAMPLE_INDICES_FILE="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --student-tm)
      STUDENT_TM="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
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
    --probe-tokens)
      PROBE_TOKENS="$2"
      shift 2
      ;;
    --trajectory-sample-index)
      TRAJECTORY_SAMPLE_INDEX="$2"
      shift 2
      ;;
    --hf-device-map)
      HF_DEVICE_MAP="$2"
      shift 2
      ;;
    --teacher-continuation-top-n)
      TEACHER_CONTINUATION_TOP_N="$2"
      shift 2
      ;;
    --teacher-continuation-max-new-tokens)
      TEACHER_CONTINUATION_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --skip-teacher-continuations)
      SKIP_TEACHER_CONTINUATIONS=1
      shift
      ;;
    --gpus|--gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$STUDENT_TM" in
  off)
    if [[ -z "$MAX_NEW_TOKENS" ]]; then
      MAX_NEW_TOKENS="1024"
    fi
    STUDENT_THINKING_ARGS=()
    ;;
  on)
    if [[ -z "$MAX_NEW_TOKENS" ]]; then
      MAX_NEW_TOKENS="16384"
    fi
    STUDENT_THINKING_ARGS=(--student-enable-thinking)
    ;;
  *)
    echo "--student-tm must be 'off' or 'on'." >&2
    exit 2
    ;;
esac

if [[ -z "$SKELETON_FILE" ]]; then
  echo "--skeleton-file is required so teacher_skeleton prompts can be reconstructed." >&2
  exit 2
fi

mkdir -p "$OUT"
read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARRAY[@]}"
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "No GPU ids configured. Set GPU_IDS or pass --gpu-ids." >&2
  exit 2
fi

MODEL_ARGS=(--base-model "$BASE_MODEL")
if [[ -n "$CHECKPOINT_DIR" ]]; then
  MODEL_ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

echo "Output directory: $OUT"
echo "Base model: $BASE_MODEL"
echo "Checkpoint: ${CHECKPOINT_DIR:-none}"
echo "Dataset: $DATASET:$SPLIT"
echo "Student TM: $STUDENT_TM | Max new tokens: $MAX_NEW_TOKENS"
echo "Sample size: $SAMPLE_SIZE | Val-N: $VAL_N"
echo "GPU ids: ${GPU_ID_ARRAY[*]} | Num shards: $NUM_SHARDS"

echo
echo "== Phase 0: 10-problem sample manifest =="
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
cp "$SKELETON_FILE" "$OUT/skeletons.jsonl"

echo
echo "== Phase 1: student-only rollout =="
pids=()
ROLLOUT_SHARD_FILES=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_rollout_openthoughts.py \
    "${MODEL_ARGS[@]}" \
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
    "${STUDENT_THINKING_ARGS[@]}" \
    --output-file "$OUT/student_rollout_shard${gpu}.jsonl" \
    --summary-file "$OUT/student_rollout_summary_shard${gpu}.json" &
  pids+=("$!")
  ROLLOUT_SHARD_FILES+=("$OUT/student_rollout_shard${gpu}.jsonl")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
ROLLOUT_MERGE_ARGS=()
for input_file in "${ROLLOUT_SHARD_FILES[@]}"; do
  ROLLOUT_MERGE_ARGS+=(--input-file "$input_file")
done
python eval/quick_jsonl_merge.py \
  "${ROLLOUT_MERGE_ARGS[@]}" \
  --output-file "$OUT/student_rollouts.jsonl"
python eval/quick_rollout_openthoughts.py \
  --summarize-only \
  --input-file "$OUT/student_rollouts.jsonl" \
  --summary-file "$OUT/student_rollout_summary.json"

echo
echo "== Phase 2: reference/skeleton teacher vs student category KL =="
pids=()
KL_SHARD_FILES=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_logit_probe.py \
    "${MODEL_ARGS[@]}" \
    --rollout-file "$OUT/student_rollouts.jsonl" \
    --skeleton-file "$OUT/skeletons.jsonl" \
    --trajectory-condition student \
    --baseline-condition student \
    --teacher-condition teacher_reference \
    --teacher-condition teacher_skeleton \
    --trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX" \
    --logit-size 0 \
    --probe-tokens "$PROBE_TOKENS" \
    --seed "$SEED" \
    --top-k "$TOP_K" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    --skip-rollout-entropy \
    --hf-device-map "$HF_DEVICE_MAP" \
    "${STUDENT_THINKING_ARGS[@]}" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --output-file "$OUT/student_teacher_category_kl_shard${gpu}.jsonl" \
    --summary-file "$OUT/student_teacher_category_kl_summary_shard${gpu}.json" &
  pids+=("$!")
  KL_SHARD_FILES+=("$OUT/student_teacher_category_kl_shard${gpu}.jsonl")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
KL_MERGE_ARGS=()
for input_file in "${KL_SHARD_FILES[@]}"; do
  KL_MERGE_ARGS+=(--input-file "$input_file")
done
python eval/quick_jsonl_merge.py \
  "${KL_MERGE_ARGS[@]}" \
  --output-file "$OUT/student_teacher_category_kl.jsonl"
python eval/quick_logit_probe.py \
  --summarize-only \
  --input-file "$OUT/student_teacher_category_kl.jsonl" \
  --summary-file "$OUT/student_teacher_category_kl_summary.json"

if [[ "$SKIP_TEACHER_CONTINUATIONS" -eq 0 ]]; then
  echo
  echo "== Phase 3: global Top-KL teacher continuations =="
  bash scripts/run_teacher_spike_continuations.sh \
    "${MODEL_ARGS[@]}" \
    --out "$OUT" \
    --gpu-ids "$GPU_IDS" \
    --top-n "$TEACHER_CONTINUATION_TOP_N" \
    --max-new-tokens "$TEACHER_CONTINUATION_MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --hf-device-map "$HF_DEVICE_MAP"
fi

echo
echo "Student-teacher category KL complete."
echo "Sample manifest:  $OUT/sample_indices.json"
echo "Student rollouts: $OUT/student_rollouts.jsonl"
echo "KL records:       $OUT/student_teacher_category_kl.jsonl"
echo "KL summary:       $OUT/student_teacher_category_kl_summary.json"
if [[ "$SKIP_TEACHER_CONTINUATIONS" -eq 0 ]]; then
  echo "Teacher report:    $OUT/visualizations/teacher_spike_continuations.html"
fi
