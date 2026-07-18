#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate an RM/Verifier LoRA checkpoint with vLLM.')
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--lora-path')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.4)
    parser.add_argument('--max-model-len', type=int, default=65536)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--max-lora-rank', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=1)
    parser.add_argument('--strict-binary-prompt', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--enforce-eager', action='store_true')
    return parser.parse_args()


STRICT_BINARY_SYSTEM = (
    'You are a strict binary verifier for software engineering trajectories. '
    'Read the issue and trajectory, decide whether the issue is resolved, and reply '
    'with exactly one token. The only valid outputs are YES or NO.'
)

STRICT_BINARY_USER_SUFFIX = (
    '\n\nFinal instruction: answer with exactly one token. '
    'Output YES if the issue is resolved, otherwise output NO. '
    'Do not output any explanation, punctuation, tags, or extra text.'
)


def build_prompt(tokenizer, messages, strict_binary_prompt: bool = False):
    if strict_binary_prompt:
        prompt_messages = []
        if messages:
            first_user_seen = False
            for message in messages:
                role = message['role']
                content = message['content']
                if role == 'system':
                    continue
                if role == 'user' and not first_user_seen:
                    content = f'{content}{STRICT_BINARY_USER_SUFFIX}'
                    first_user_seen = True
                prompt_messages.append({'role': role, 'content': content})
            if not first_user_seen:
                prompt_messages.append({'role': 'user', 'content': STRICT_BINARY_USER_SUFFIX.strip()})
        else:
            prompt_messages = [{'role': 'user', 'content': STRICT_BINARY_USER_SUFFIX.strip()}]
        prompt_messages.insert(0, {'role': 'system', 'content': STRICT_BINARY_SYSTEM})
    else:
        prompt_messages = messages

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def normalize_prediction(text: str) -> int | None:
    value = text.strip().upper()
    if value.startswith('YES'):
        return 1
    if value.startswith('NO'):
        return 0
    return None


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    rows = []
    with data_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    prompts = [
        build_prompt(
            tokenizer,
            row['messages'][:-1],
            strict_binary_prompt=args.strict_binary_prompt,
        )
        for row in rows
    ]

    enable_lora = bool(args.lora_path)

    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_lora=enable_lora,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
    )
    lora_request = LoRARequest('checkpoint', 1, args.lora_path) if enable_lora else None

    records = []
    correct = 0
    parsed = 0
    for idx, output in enumerate(
        llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_request)
    ):
        text = output.outputs[0].text
        pred = normalize_prediction(text)
        label = int(rows[idx]['rm_label'])
        if pred is not None:
            parsed += 1
            correct += int(pred == label)
        record = {
            'index': idx,
            'instance_id': rows[idx].get('instance_id'),
            'traj_id': rows[idx].get('traj_id'),
            'label': label,
            'pred': pred,
            'raw_text': text,
        }
        records.append(record)
        if args.save_every > 0 and ((idx + 1) % args.save_every == 0):
            output_path.write_text(
                json.dumps(
                    {
                        'num_examples': len(rows),
                        'completed': idx + 1,
                        'parsed': parsed,
                        'correct': correct,
                        'accuracy': (correct / parsed) if parsed else None,
                        'records': records,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    output_path.write_text(
        json.dumps(
            {
                'num_examples': len(rows),
                'completed': len(rows),
                'parsed': parsed,
                'correct': correct,
                'accuracy': (correct / parsed) if parsed else None,
                'records': records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                'num_examples': len(rows),
                'parsed': parsed,
                'correct': correct,
                'accuracy': (correct / parsed) if parsed else None,
                'output_path': str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
