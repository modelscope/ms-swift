"""HF GRPO multimodal smoke test.

Replaces test_mllm_pt's coco_2014_caption (incompatible with datasets 5.x)
with AI-ModelScope/clevr_cogen_a_train (no script-based loading).

Validates: deletion of TRL super delegation; local_forward path on multimodal
inputs; collate handling of vision-related kwargs.

Lightweight: Qwen2-VL-2B-Instruct + LoRA + 2 steps + 20 samples.
"""
import os

from swift import RLHFArguments, rlhf_main

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

SYSTEM_PROMPT = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
                 'The assistant first thinks about the reasoning process in the mind and then provides the user '
                 'with the answer. The reasoning process and answer are enclosed within <think> </think> '
                 'and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> '
                 'answer here </answer>')


def main():
    rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2-VL-2B-Instruct',
            tuner_type='lora',
            dataset=['AI-ModelScope/clevr_cogen_a_train#20'],
            system=SYSTEM_PROMPT,
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
        ))


if __name__ == '__main__':
    main()
