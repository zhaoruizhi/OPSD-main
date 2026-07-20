#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-/data0/shared/Qwen3-1.7B}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUT="${OUT:-}"
KL_FILE="${KL_FILE:-}"
STUDENT_ROLLOUT_FILE="${STUDENT_ROLLOUT_FILE:-}"
SKELETON_FILE="${SKELETON_FILE:-}"
GPU_IDS="${GPU_IDS:-4}"
TOP_N="${TOP_N:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20000}"
HF_DEVICE_MAP="${HF_DEVICE_MAP:-cuda}"

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
    --out)
      OUT="$2"
      shift 2
      ;;
    --kl-file)
      KL_FILE="$2"
      shift 2
      ;;
    --student-rollout-file)
      STUDENT_ROLLOUT_FILE="$2"
      shift 2
      ;;
    --skeleton-file)
      SKELETON_FILE="$2"
      shift 2
      ;;
    --gpus|--gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --top-n)
      TOP_N="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --max-model-len|--max-context-tokens)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --hf-device-map)
      HF_DEVICE_MAP="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT" ]]; then
  echo "--out is required and must point to a completed student-teacher category-KL directory." >&2
  exit 2
fi
if [[ ! -d "$OUT" ]]; then
  echo "Output directory does not exist: $OUT" >&2
  exit 2
fi

if [[ -z "$STUDENT_ROLLOUT_FILE" ]]; then
  STUDENT_ROLLOUT_FILE="$OUT/student_rollouts.jsonl"
fi
if [[ -z "$SKELETON_FILE" ]]; then
  SKELETON_FILE="$OUT/skeletons.jsonl"
fi
if [[ ! -f "$STUDENT_ROLLOUT_FILE" ]]; then
  echo "Student rollout file does not exist: $STUDENT_ROLLOUT_FILE" >&2
  exit 2
fi
if [[ ! -f "$SKELETON_FILE" ]]; then
  echo "Skeleton file does not exist: $SKELETON_FILE" >&2
  exit 2
fi

read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARRAY[@]}"
if [[ "$NUM_SHARDS" -eq 0 ]]; then
  echo "No GPU ids configured. Pass --gpu-ids \"0 1\" or another space-separated list." >&2
  exit 2
fi

MODEL_ARGS=(--base-model "$BASE_MODEL")
if [[ -n "$CHECKPOINT_DIR" ]]; then
  MODEL_ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi

echo "Output directory: $OUT"
echo "Base model: $BASE_MODEL"
echo "Checkpoint: ${CHECKPOINT_DIR:-none}"
echo "GPU ids: ${GPU_ID_ARRAY[*]} | Num workers: $NUM_SHARDS"
echo "Global Top N: $TOP_N | Greedy continuation tokens: $MAX_NEW_TOKENS"

echo
echo "== Phase C0: validate/rebuild complete KL aggregate =="
KL_REMERGED_FILE="$OUT/student_teacher_category_kl_remerged.jsonl"
KL_INPUT_FILES=()
if [[ -n "$KL_FILE" ]]; then
  if [[ ! -f "$KL_FILE" ]]; then
    echo "KL file does not exist: $KL_FILE" >&2
    exit 2
  fi
  KL_INPUT_FILES+=("$KL_FILE")
else
  shopt -s nullglob
  KL_SHARD_FILES=("$OUT"/student_teacher_category_kl_shard*.jsonl)
  shopt -u nullglob
  if [[ "${#KL_SHARD_FILES[@]}" -gt 0 ]]; then
    KL_INPUT_FILES+=("${KL_SHARD_FILES[@]}")
  elif [[ -f "$OUT/student_teacher_category_kl.jsonl" ]]; then
    KL_INPUT_FILES+=("$OUT/student_teacher_category_kl.jsonl")
  else
    echo "No KL shards or aggregate found under $OUT" >&2
    exit 2
  fi
fi

KL_MERGE_ARGS=()
for input_file in "${KL_INPUT_FILES[@]}"; do
  KL_MERGE_ARGS+=(--input-file "$input_file")
done
python eval/quick_jsonl_merge.py \
  "${KL_MERGE_ARGS[@]}" \
  --output-file "$KL_REMERGED_FILE"

echo
echo "== Phase C1: generate teacher continuations at global KL spikes =="
pids=()
CONTINUATION_SHARD_FILES=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  shard_file="$OUT/teacher_spike_continuation_shard${gpu}.jsonl"
  CONTINUATION_SHARD_FILES+=("$shard_file")
  CUDA_VISIBLE_DEVICES="$gpu" python eval/quick_teacher_spike_continuation.py \
    "${MODEL_ARGS[@]}" \
    --kl-file "$KL_REMERGED_FILE" \
    --student-rollout-file "$STUDENT_ROLLOUT_FILE" \
    --skeleton-file "$SKELETON_FILE" \
    --top-n "$TOP_N" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    --hf-device-map "$HF_DEVICE_MAP" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --output-file "$shard_file" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo
echo "== Phase C2: merge and render three-column report =="
CONTINUATION_MERGE_ARGS=()
for input_file in "${CONTINUATION_SHARD_FILES[@]}"; do
  CONTINUATION_MERGE_ARGS+=(--input-file "$input_file")
done
CONTINUATION_FILE="$OUT/teacher_spike_continuations.jsonl"
python eval/quick_jsonl_merge.py \
  "${CONTINUATION_MERGE_ARGS[@]}" \
  --output-file "$CONTINUATION_FILE" \
  --sort-key rank

SUMMARY_FILE="$OUT/teacher_spike_continuation_summary.json"
REPORT_FILE="$OUT/visualizations/teacher_spike_continuations.html"
python eval/quick_teacher_spike_continuation.py \
  --render-only \
  --input-file "$CONTINUATION_FILE" \
  --summary-file "$SUMMARY_FILE" \
  --report-file "$REPORT_FILE"

echo
echo "Teacher KL spike continuations complete."
echo "Validated KL records: $KL_REMERGED_FILE"
echo "Continuation records: $CONTINUATION_FILE"
echo "Summary:              $SUMMARY_FILE"
echo "HTML report:          $REPORT_FILE"
