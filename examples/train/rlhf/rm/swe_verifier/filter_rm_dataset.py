#!/usr/bin/env python3
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER = None


def init_worker(model_path: str) -> None:
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def count_tokens(messages) -> int:
    input_ids = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    return len(input_ids)


def rough_char_len(messages) -> int:
    total = 0
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, str):
            total += len(content)
        else:
            total += len(json.dumps(content, ensure_ascii=False))
    return total


def process_row(args):
    line, max_length, char_guard_factor = args
    row = json.loads(line)
    char_len = rough_char_len(row['messages'])
    char_guard = max_length * char_guard_factor
    if char_len > char_guard:
        return row, None, char_len, False
    token_len = count_tokens(row['messages'])
    return row, token_len, char_len, token_len <= max_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--num-proc', type=int, default=8)
    parser.add_argument('--char-guard-factor', type=int, default=12)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    max_seen = 0
    max_chars = 0

    with input_path.open('r', encoding='utf-8') as fin:
        lines = fin.readlines()

    with output_path.open('w', encoding='utf-8') as fout:
        with Pool(processes=args.num_proc, initializer=init_worker, initargs=(args.model_path,)) as pool:
            work_items = ((line, args.max_length, args.char_guard_factor) for line in lines)
            for idx, (row, token_len, char_len, keep) in enumerate(pool.imap(process_row, work_items, chunksize=16), 1):
                if token_len is not None:
                    row['token_length'] = token_len
                    max_seen = max(max_seen, token_len)
                max_chars = max(max_chars, char_len)
                if keep:
                    fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                    kept += 1
                else:
                    dropped += 1

                if idx % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                'processed': idx,
                                'kept': kept,
                                'dropped': dropped,
                                'max_seen': max_seen,
                                'max_chars': max_chars,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    print(
        json.dumps(
            {
                'input': str(input_path),
                'output': str(output_path),
                'max_length': args.max_length,
                'kept': kept,
                'dropped': dropped,
                'max_seen': max_seen,
                'max_chars': max_chars,
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
