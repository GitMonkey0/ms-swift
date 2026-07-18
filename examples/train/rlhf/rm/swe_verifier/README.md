# SWE Verifier RM

This example trains a trajectory-level SWE verifier / reward model with LoRA on top of a Qwen3.6-style base model using `swift sft` and DeepSpeed ZeRO-3.

It is intended for binary verification tasks where the model must decide whether an agent trajectory resolved the issue, and where inference-time evaluation is done with `vLLM`.

## Files

- `train_qwen36_rm_transformers_deepspeed.sh`: training launcher
- `filter_rm_dataset.py`: optional max-length filtering pass before training
- `eval_vllm_lora.py`: offline `vLLM` evaluation script for base model and LoRA checkpoints

## Expected dataset format

Training and evaluation files are JSONL. Each row is expected to contain:

- `messages`: chat messages including the verifier prompt and labeled answer
- `rm_label`: integer label, `1` for `YES`, `0` for `NO`
- optional metadata such as `instance_id` and `traj_id`

The evaluator uses `row["messages"][:-1]` as the prompt and `rm_label` as the target.

## Training

Example:

```bash
MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
RAW_TRAIN_DATA=/path/to/train.jsonl \
RAW_EVAL_DATA=/path/to/eval.jsonl \
OUTPUT_DIR=/path/to/output/qwen36-rm \
bash examples/train/rlhf/rm/swe_verifier/train_qwen36_rm_transformers_deepspeed.sh
```

Important defaults in the launcher:

- LoRA on `all-linear all-router`
- DeepSpeed `zero3`
- `max_length=65536`
- `attn_impl=flash_attn`
- `group_by_length=true`
- `loss_scale=last_round`
- `save_steps=200`
- `save_total_limit=2`

## Gradient checkpointing

For the `ms-swift + PEFT LoRA + DeepSpeed ZeRO-3` setup used here, the working configuration is:

- `--gradient_checkpointing true`
- `--gradient_checkpointing_kwargs '{"use_reentrant": true}'`

This example also depends on the local `ms-swift` patch in `swift/pipelines/train/tuner.py` that calls:

```python
Swift.prepare_model(model, lora_config, autocast_adapter_dtype=False)
```

That avoids the dtype mismatch seen with ZeRO-3 + LoRA in this setup.

## Evaluation

Use strict binary prompting when comparing the base model and LoRA checkpoints:

- `--strict-binary-prompt`

This forces the outer instruction to require exactly one token, `YES` or `NO`, which makes base-model and checkpoint comparisons much more stable.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train/rlhf/rm/swe_verifier/eval_vllm_lora.py \
  --base-model /path/to/Qwen3.6-35B-A3B \
  --lora-path /path/to/checkpoint-2000 \
  --data-path /path/to/eval.jsonl \
  --output-path /path/to/checkpoint-2000-strict-eval100.json \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-tokens 1 \
  --strict-binary-prompt \
  --limit 100 \
  --enforce-eager
```

Example for evaluating the base model under the same prompt contract:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/train/rlhf/rm/swe_verifier/eval_vllm_lora.py \
  --base-model /path/to/Qwen3.6-35B-A3B \
  --data-path /path/to/eval.jsonl \
  --output-path /path/to/base-strict-eval100.json \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-tokens 1 \
  --strict-binary-prompt \
  --limit 100 \
  --enforce-eager
```
