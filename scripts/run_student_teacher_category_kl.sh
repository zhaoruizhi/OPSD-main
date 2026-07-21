#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-${MODEL:-/data0/shared/Qwen3-1.7B}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
SPLIT="${SPLIT:-train}"
OUT="${OUT:-/data1/opsd_quick/qwen31b_student_teacher_category_kl_$(date +%Y%m%d_%H%M%S)}"
SKELETON_FILE="${SKELETON_FILE:-}"
SAMPLE_INDICES_FILE="${SAMPLE_INDICES_FILE:-}"
SAMPLE_SIZE=10
SAMPLE_SIZE_EXPLICIT=0
VAL_N="${VAL_N:-4}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-current-style-neutral}"
TARGET_TOKEN_SOURCE="${TARGET_TOKEN_SOURCE:-}"
STUDENT_TM="${STUDENT_TM:-off}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
STUDENT_MAX_NEW_TOKENS="${STUDENT_MAX_NEW_TOKENS:-}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-}"
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
      SAMPLE_SIZE_EXPLICIT=1
      shift 2
      ;;
    --val-n)
      VAL_N="$2"
      shift 2
      ;;
    --experiment-profile)
      EXPERIMENT_PROFILE="$2"
      shift 2
      ;;
    --target-token-source)
      TARGET_TOKEN_SOURCE="$2"
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
    --student-max-new-tokens)
      STUDENT_MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --teacher-max-new-tokens)
      TEACHER_MAX_NEW_TOKENS="$2"
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

case "$EXPERIMENT_PROFILE" in
  current-style-neutral)
    TARGET_TOKEN_SOURCE="${TARGET_TOKEN_SOURCE:-auto}"
    ;;
  legacy-20260629)
    TARGET_TOKEN_SOURCE="${TARGET_TOKEN_SOURCE:-target_tail_text}"
    if [[ "$SAMPLE_SIZE_EXPLICIT" -eq 0 ]]; then
      SAMPLE_SIZE=128
    fi
    if [[ "$STUDENT_TM" != "off" ]]; then
      echo "legacy-20260629 requires --student-tm off." >&2
      exit 2
    fi
    ;;
  *)
    echo "--experiment-profile must be 'current-style-neutral' or 'legacy-20260629'." >&2
    exit 2
    ;;
esac

case "$TARGET_TOKEN_SOURCE" in
  auto|target_tail_text)
    ;;
  *)
    echo "--target-token-source must be 'auto' or 'target_tail_text'." >&2
    exit 2
    ;;
esac

case "$STUDENT_TM" in
  off)
    if [[ "$EXPERIMENT_PROFILE" == "legacy-20260629" ]]; then
      DEFAULT_STUDENT_MAX_NEW_TOKENS="16384"
    else
      DEFAULT_STUDENT_MAX_NEW_TOKENS="1024"
    fi
    STUDENT_THINKING_ARGS=()
    ;;
  on)
    DEFAULT_STUDENT_MAX_NEW_TOKENS="16384"
    STUDENT_THINKING_ARGS=(--student-enable-thinking)
    ;;
  *)
    echo "--student-tm must be 'off' or 'on'." >&2
    exit 2
    ;;
esac

if [[ -z "$STUDENT_MAX_NEW_TOKENS" ]]; then
  STUDENT_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$DEFAULT_STUDENT_MAX_NEW_TOKENS}"
fi
if [[ -z "$TEACHER_MAX_NEW_TOKENS" ]]; then
  TEACHER_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
fi

validate_positive_integer() {
  local label="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$label must be a positive integer, got: $value" >&2
    exit 2
  fi
}

validate_positive_integer "student max new tokens" "$STUDENT_MAX_NEW_TOKENS"
validate_positive_integer "teacher max new tokens" "$TEACHER_MAX_NEW_TOKENS"
validate_positive_integer "max model length" "$MAX_MODEL_LEN"
validate_positive_integer "sample size" "$SAMPLE_SIZE"
validate_positive_integer "val n" "$VAL_N"

if [[ -z "$SKELETON_FILE" ]]; then
  echo "--skeleton-file is required so teacher_skeleton prompts can be reconstructed." >&2
  exit 2
fi
if [[ "$EXPERIMENT_PROFILE" == "legacy-20260629" && -z "$SAMPLE_INDICES_FILE" ]]; then
  echo "legacy-20260629 requires --sample-indices-file from the archived experiment." >&2
  exit 2
fi
if [[ ! -f "$SKELETON_FILE" ]]; then
  echo "Skeleton file does not exist: $SKELETON_FILE" >&2
  exit 2
fi
if [[ -n "$SAMPLE_INDICES_FILE" && ! -f "$SAMPLE_INDICES_FILE" ]]; then
  echo "Sample indices file does not exist: $SAMPLE_INDICES_FILE" >&2
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
echo "Experiment profile: $EXPERIMENT_PROFILE | KL target token source: $TARGET_TOKEN_SOURCE"
echo "Student TM: $STUDENT_TM | Student max new tokens: $STUDENT_MAX_NEW_TOKENS"
echo "Teacher max new tokens: $TEACHER_MAX_NEW_TOKENS"
echo "Model context length: $MAX_MODEL_LEN"
echo "Sample size: $SAMPLE_SIZE | Val-N: $VAL_N"
echo "GPU ids: ${GPU_ID_ARRAY[*]} | Num shards: $NUM_SHARDS"

echo
echo "== Phase 0: shared sample manifest =="
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

if [[ "$SKIP_TEACHER_CONTINUATIONS" -eq 0 ]]; then
  TEACHER_CONTINUATIONS_STATE="enabled"
else
  TEACHER_CONTINUATIONS_STATE="skipped"
fi
python eval/write_experiment_config.py \
  --output-file "$OUT/experiment_config.json" \
  --repo-root . \
  --experiment-profile "$EXPERIMENT_PROFILE" \
  --base-model "$BASE_MODEL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --sample-indices-file "$OUT/sample_indices.json" \
  --skeleton-file "$OUT/skeletons.jsonl" \
  --sample-size "$SAMPLE_SIZE" \
  --val-n "$VAL_N" \
  --student-tm "$STUDENT_TM" \
  --student-max-new-tokens "$STUDENT_MAX_NEW_TOKENS" \
  --teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --seed "$SEED" \
  --gpu-ids "$GPU_IDS" \
  --trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX" \
  --probe-tokens "$PROBE_TOKENS" \
  --target-token-source "$TARGET_TOKEN_SOURCE" \
  --hf-device-map "$HF_DEVICE_MAP" \
  --teacher-continuations "$TEACHER_CONTINUATIONS_STATE"

echo
echo "== Phase 1: four-condition rollouts =="
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
    --skeleton-file "$OUT/skeletons.jsonl" \
    --teacher-prompt-profile "$EXPERIMENT_PROFILE" \
    --seed "$SEED" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --val-n "$VAL_N" \
    --student-max-new-tokens "$STUDENT_MAX_NEW_TOKENS" \
    --teacher-max-new-tokens "$TEACHER_MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${STUDENT_THINKING_ARGS[@]}" \
    --output-file "$OUT/rollout_shard${gpu}.jsonl" \
    --summary-file "$OUT/rollout_summary_shard${gpu}.json" &
  pids+=("$!")
  ROLLOUT_SHARD_FILES+=("$OUT/rollout_shard${gpu}.jsonl")
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
  --output-file "$OUT/rollouts.jsonl"
python eval/quick_rollout_openthoughts.py \
  --summarize-only \
  --input-file "$OUT/rollouts.jsonl" \
  --summary-file "$OUT/rollout_summary.json"

echo
echo "== Phase 2: reference/skeleton teacher vs teacher_base KL and rollout entropy =="
pids=()
BASE_KL_SHARD_FILES=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_logit_probe.py \
    "${MODEL_ARGS[@]}" \
    --rollout-file "$OUT/rollouts.jsonl" \
    --skeleton-file "$OUT/skeletons.jsonl" \
    --teacher-prompt-profile "$EXPERIMENT_PROFILE" \
    --trajectory-condition teacher_base \
    --baseline-condition teacher_base \
    --teacher-condition teacher_reference \
    --teacher-condition teacher_skeleton \
    --trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX" \
    --logit-size 0 \
    --probe-tokens "$PROBE_TOKENS" \
    --target-token-source "$TARGET_TOKEN_SOURCE" \
    --seed "$SEED" \
    --top-k "$TOP_K" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    --require-context-rollouts \
    --hf-device-map "$HF_DEVICE_MAP" \
    "${STUDENT_THINKING_ARGS[@]}" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --output-file "$OUT/logit_probe_shard${gpu}.jsonl" \
    --summary-file "$OUT/logit_summary_shard${gpu}.json" &
  pids+=("$!")
  BASE_KL_SHARD_FILES+=("$OUT/logit_probe_shard${gpu}.jsonl")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
BASE_KL_MERGE_ARGS=()
for input_file in "${BASE_KL_SHARD_FILES[@]}"; do
  BASE_KL_MERGE_ARGS+=(--input-file "$input_file")
done
python eval/quick_jsonl_merge.py \
  "${BASE_KL_MERGE_ARGS[@]}" \
  --output-file "$OUT/logit_probe.jsonl"
python eval/quick_logit_probe.py \
  --summarize-only \
  --input-file "$OUT/logit_probe.jsonl" \
  --summary-file "$OUT/logit_summary.json"

echo
echo "== Phase 2b: teacher_base KL comparison report =="
python eval/quick_teacher_base_kl_report.py \
  --logit-file "$OUT/logit_probe.jsonl" \
  --rollout-file "$OUT/rollouts.jsonl" \
  --rollout-summary-file "$OUT/rollout_summary.json" \
  --skeleton-file "$OUT/skeletons.jsonl" \
  --csv-file "$OUT/visualizations/teacher_base_kl_reference_vs_skeleton_top_spikes.csv" \
  --spikes-jsonl-file "$OUT/visualizations/teacher_base_top_distribution_spikes.jsonl" \
  --report-file "$OUT/visualizations/teacher_base_kl_reference_vs_skeleton_report.html"

echo
echo "== Phase 3: reference/skeleton teacher vs student category KL =="
pids=()
STUDENT_KL_SHARD_FILES=()
for gpu_index in "${!GPU_ID_ARRAY[@]}"; do
  gpu="${GPU_ID_ARRAY[$gpu_index]}"
  shard_id="$gpu_index"
  CUDA_VISIBLE_DEVICES=$gpu python eval/quick_logit_probe.py \
    "${MODEL_ARGS[@]}" \
    --rollout-file "$OUT/rollouts.jsonl" \
    --skeleton-file "$OUT/skeletons.jsonl" \
    --teacher-prompt-profile "$EXPERIMENT_PROFILE" \
    --trajectory-condition student \
    --baseline-condition student \
    --teacher-condition teacher_reference \
    --teacher-condition teacher_skeleton \
    --trajectory-sample-index "$TRAJECTORY_SAMPLE_INDEX" \
    --logit-size 0 \
    --probe-tokens "$PROBE_TOKENS" \
    --target-token-source "$TARGET_TOKEN_SOURCE" \
    --seed "$SEED" \
    --top-k "$TOP_K" \
    --max-context-tokens "$MAX_MODEL_LEN" \
    --require-context-rollouts \
    --skip-rollout-entropy \
    --hf-device-map "$HF_DEVICE_MAP" \
    "${STUDENT_THINKING_ARGS[@]}" \
    --shard-id "$shard_id" \
    --num-shards "$NUM_SHARDS" \
    --output-file "$OUT/student_teacher_category_kl_shard${gpu}.jsonl" \
    --summary-file "$OUT/student_teacher_category_kl_summary_shard${gpu}.json" &
  pids+=("$!")
  STUDENT_KL_SHARD_FILES+=("$OUT/student_teacher_category_kl_shard${gpu}.jsonl")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
STUDENT_KL_MERGE_ARGS=()
for input_file in "${STUDENT_KL_SHARD_FILES[@]}"; do
  STUDENT_KL_MERGE_ARGS+=(--input-file "$input_file")
done
python eval/quick_jsonl_merge.py \
  "${STUDENT_KL_MERGE_ARGS[@]}" \
  --output-file "$OUT/student_teacher_category_kl.jsonl"
python eval/quick_logit_probe.py \
  --summarize-only \
  --input-file "$OUT/student_teacher_category_kl.jsonl" \
  --summary-file "$OUT/student_teacher_category_kl_summary.json"

if [[ "$SKIP_TEACHER_CONTINUATIONS" -eq 0 ]]; then
  echo
  echo "== Phase 4: global Top-KL teacher continuations =="
  bash scripts/run_teacher_spike_continuations.sh \
    "${MODEL_ARGS[@]}" \
    --out "$OUT" \
    --kl-file "$OUT/student_teacher_category_kl.jsonl" \
    --student-rollout-file "$OUT/rollouts.jsonl" \
    --teacher-prompt-profile "$EXPERIMENT_PROFILE" \
    --gpu-ids "$GPU_IDS" \
    --top-n "$TEACHER_CONTINUATION_TOP_N" \
    --max-new-tokens "$TEACHER_CONTINUATION_MAX_NEW_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --hf-device-map "$HF_DEVICE_MAP"
fi

echo
echo "Dual-KL semantic-skeleton ablation complete."
echo "Sample manifest:       $OUT/sample_indices.json"
echo "Experiment config:      $OUT/experiment_config.json"
echo "Four-condition rollout: $OUT/rollouts.jsonl"
echo "Performance/token length: $OUT/rollout_summary.json"
echo "Teacher-base KL records: $OUT/logit_probe.jsonl"
echo "Teacher-base KL summary: $OUT/logit_summary.json"
echo "Teacher-base KL report:  $OUT/visualizations/teacher_base_kl_reference_vs_skeleton_report.html"
echo "Student-target KL records: $OUT/student_teacher_category_kl.jsonl"
echo "Student-target KL summary: $OUT/student_teacher_category_kl_summary.json"
if [[ "$SKIP_TEACHER_CONTINUATIONS" -eq 0 ]]; then
  echo "Teacher continuation report: $OUT/visualizations/teacher_spike_continuations.html"
fi
