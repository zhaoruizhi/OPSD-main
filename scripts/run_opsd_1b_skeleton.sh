#!/usr/bin/env bash
set -euo pipefail

SKELETON_FILE="${SKELETON_FILE:-/home/ruizzhao/OPSD-main/outputs/opsd_skeletons/qwen31b_full_train_20260703_130644/skeletons.jsonl}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12949}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROCESSES" \
    --gradient_accumulation_steps 2 \
    --main_process_port "$MAIN_PROCESS_PORT" \
    opsd_train.py \
    --model_name_or_path /home/ruizzhao/OPSD-main/models/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir  /home/ruizzhao/OPSD-main/outputs/opsd/ \
    --run_config qwen31b_gen1024_skeleton_fixteacher_temp11_forwardbeta0_clip005 \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --teacher_context_mode skeleton \
    --skeleton_file "$SKELETON_FILE" \
    --skeleton_subset_policy error \
    --report_to wandb \
    --wandb_project OPSD
