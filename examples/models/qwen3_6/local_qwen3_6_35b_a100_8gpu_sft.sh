#!/usr/bin/env bash
set -euo pipefail

# 8 x A100 LoRA SFT for a local Qwen3.6-35B-A3B model.
# Run from the ms-swift repo root or anywhere after adjusting the variables below.

ROOT_DIR="${ROOT_DIR:-/opt/tiger/sft}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/Qwen3.6-35B-A3B}"

# Replace with a real local dataset path or a registered dataset id before running.
DATASET="${DATASET:-/path/to/your_sft_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/Qwen3.6-35B-A3B-lora}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "model config not found: ${MODEL_DIR}/config.json" >&2
    exit 1
fi

if [[ "${DATASET}" == "/path/to/your_sft_dataset.jsonl" ]]; then
    echo "Please set DATASET to a real dataset path or dataset id before running." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

python swift/cli/main.py sft \
    --model "${MODEL_DIR}" \
    --model_type qwen3_5_moe \
    --template qwen3_5 \
    --tuner_type lora \
    --dataset "${DATASET}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attention_2 \
    --experts_impl grouped_mm \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-3 \
    --group_by_length true \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed zero3
