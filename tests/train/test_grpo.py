import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
}

SYSTEM_PROMPT = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
                 'The assistant first thinks about the reasoning process in the mind and then provides the user '
                 'with the answer. The reasoning process and answer are enclosed within <think> </think> '
                 'and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> '
                 'answer here </answer>')


def test_llm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-1.5B-Instruct',
            train_type='full',
            dataset=['AI-MO/NuminaMath-TIR#100'],
            split_dataset_ratio=0.1,
            system=SYSTEM_PROMPT,
            reward_funcs=['accuracy', 'format'],
            max_completion_length=4096,
            num_generations=2,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_llm_zero2():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-1.5B-Instruct',
            train_type='full',
            dataset=['AI-MO/NuminaMath-TIR#100'],
            system=SYSTEM_PROMPT,
            reward_funcs=['accuracy', 'format'],
            max_completion_length=4096,
            num_generations=2,
            deepspeed='zero2',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_llm_vllm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-1.5B-Instruct',
            reward_model='AI-ModelScope/GRM_Llama3.1_8B_rewardmodel-ft',
            train_type='full',
            dataset=['AI-MO/NuminaMath-TIR#100'],
            system=SYSTEM_PROMPT,
            reward_funcs=['accuracy', 'format'],
            use_vllm=True,
            max_completion_length=4096,
            num_generations=2,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_llm_vllm_zero2():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2.5-1.5B-Instruct',
            train_type='full',
            dataset=['AI-MO/NuminaMath-TIR#100'],
            system=SYSTEM_PROMPT,
            reward_funcs=['accuracy', 'format'],
            use_vllm=True,
            max_completion_length=4096,
            num_generations=2,
            deepspeed='zero2',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_pt():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2-VL-2B-Instruct',
            train_type='full',
            # dataset=['AI-MO/NuminaMath-TIR#100'],
            dataset=['modelscope/coco_2014_caption:validation#100'],
            system=SYSTEM_PROMPT,
            reward_funcs=['format'],
            max_completion_length=4096,
            num_generations=2,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    # test_llm_zero3()
    # test_llm_vllm()
    # test_llm_vllm_zero2()
    test_mllm_pt()
