"""HF GRPO padding_free=True smoke test.

Validates the only non-trivial branch in collate_to_micro_batch + region_frame:
the first-row seq_lengths adjustment in build_completion_mask_and_seq_lengths
when padding_free=True (rmpad/packing).

Lightweight: Qwen2-0.5B + LoRA + 2 steps + 20 samples + completion_length=128.
"""
import os

from swift import RLHFArguments, rlhf_main

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2-0.5B',
            tuner_type='lora',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
            reward_funcs=['format'],
            max_completion_length=128,
            num_generations=2,
            max_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=2,
            split_dataset_ratio=0.01,
            logging_steps=1,
            use_vllm=False,
            eval_strategy='no',
            padding_free=True,
            attn_impl='flash_attn',
        ))


if __name__ == '__main__':
    main()
