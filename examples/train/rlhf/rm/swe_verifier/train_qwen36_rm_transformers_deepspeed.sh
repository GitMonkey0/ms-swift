#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NPROC_PER_NODE:=8}"
: "${MODEL_PATH:=/path/to/Qwen3.6-35B-A3B}"
: "${RAW_TRAIN_DATA:=/path/to/train.jsonl}"
: "${RAW_EVAL_DATA:=/path/to/eval.jsonl}"
: "${DISABLE_EVAL:=0}"
: "${DISABLE_GRADIENT_CHECKPOINTING:=0}"
: "${OUTPUT_DIR:=${REPO_ROOT}/output/qwen36-rm-gencls-ds-128k}"
: "${MAX_LENGTH:=65536}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=1}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=1}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${NUM_EPOCHS:=1}"
: "${LR:=1e-4}"
: "${WARMUP_RATIO:=0.05}"
: "${SAVE_STEPS:=200}"
: "${EVAL_STEPS:=1000}"
: "${LOGGING_STEPS:=5}"
: "${SAVE_TOTAL_LIMIT:=2}"
: "${LORA_RANK:=8}"
: "${LORA_ALPHA:=32}"
: "${DEEPSPEED_STAGE:=zero3}"
: "${DATASET_NUM_PROC:=8}"
: "${DATALOADER_NUM_WORKERS:=8}"
: "${TORCH_DTYPE:=bfloat16}"
: "${LOSS_SCALE:=last_round}"
: "${FILTER_DATASET:=1}"

TRAIN_DATA="${RAW_TRAIN_DATA}"
EVAL_DATA="${RAW_EVAL_DATA}"
FILTERED_DATA_DIR="$(dirname "${RAW_TRAIN_DATA}")/filtered-${MAX_LENGTH}"
FILTERED_TRAIN_DATA="${FILTERED_DATA_DIR}/train.jsonl"
FILTERED_EVAL_DATA="${FILTERED_DATA_DIR}/eval.jsonl"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export PATH="$HOME/.local/bin:$PATH"

if [[ "${FILTER_DATASET}" == "1" ]]; then
  mkdir -p "${FILTERED_DATA_DIR}"
  if [[ ! -f "${FILTERED_TRAIN_DATA}" ]]; then
    python "${SCRIPT_DIR}/filter_rm_dataset.py" \
      --input "${RAW_TRAIN_DATA}" \
      --output "${FILTERED_TRAIN_DATA}" \
      --model-path "${MODEL_PATH}" \
      --max-length "${MAX_LENGTH}"
  fi
  if [[ ! -f "${FILTERED_EVAL_DATA}" ]]; then
    python "${SCRIPT_DIR}/filter_rm_dataset.py" \
      --input "${RAW_EVAL_DATA}" \
      --output "${FILTERED_EVAL_DATA}" \
      --model-path "${MODEL_PATH}" \
      --max-length "${MAX_LENGTH}"
  fi
  TRAIN_DATA="${FILTERED_TRAIN_DATA}"
  EVAL_DATA="${FILTERED_EVAL_DATA}"
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(
    python - <<'PY'
import socket
s = socket.socket()
s.bind(('0.0.0.0', 0))
print(s.getsockname()[1])
s.close()
PY
  )"
fi
export MASTER_PORT

swift_args=(
  --model "${MODEL_PATH}" \
  --task_type causal_lm \
  --dataset "${TRAIN_DATA}" \
  --load_from_cache_file true \
  --split_dataset_ratio 0 \
  --torch_dtype "${TORCH_DTYPE}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear all-router \
  --deepspeed "${DEEPSPEED_STAGE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --max_length "${MAX_LENGTH}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --loss_scale "${LOSS_SCALE}" \
  --group_by_length true \
  --attn_impl flash_attn \
  --experts_impl grouped_mm \
  --router_aux_loss_coef 1e-6
)

if [[ "${DISABLE_EVAL}" != "1" ]]; then
  swift_args+=(
    --val_dataset "${EVAL_DATA}" \
  )
fi

if [[ "${DISABLE_GRADIENT_CHECKPOINTING}" != "1" ]]; then
  swift_args+=(
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
  )
else
  swift_args+=(
    --gradient_checkpointing false \
  )
fi

swift sft "${swift_args[@]}"
