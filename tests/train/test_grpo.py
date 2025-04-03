import os

from swift.llm import InferArguments, RLHFArguments, infer_main, rlhf_main

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

SYSTEM_PROMPT = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
                 'The assistant first thinks about the reasoning process in the mind and then provides the user '
                 'with the answer. The reasoning process and answer are enclosed within <think> </think> '
                 'and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> '
                 'answer here </answer>')

kwargs = {
    'rlhf_type': 'grpo',
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
    'num_generations': 2,
    'max_completion_length': 2048,
    'max_steps': 10,
    'eval_steps': 5,
    'system': SYSTEM_PROMPT,
}


def test_llm_pt():
    os.environ['NPROC_PER_NODES'] = 4
    multi_turn_funcs = [None, 'math_tip_trick']
    for multi_turn_func in multi_turn_funcs:
        kwargs['multi_turn_func'] = multi_turn_func
        rlhf_main(
            RLHFArguments(
                model='Qwen/Qwen2.5-1.5B-Instruct',
                dataset=['MATH-lighteval#100'],
                reward_funcs=['accuracy', 'format'],
                **kwargs))


def test_llm_vllm():
    tp_sizes = [1, 4]
    infer_workers = [1, 4]  # async_mode colocate_mode
    multi_turn_funcs = [None, 'math_tip_trick']
    for num_infer_workers in infer_workers:
        for tp_size in tp_sizes:
            for multi_turn_func in multi_turn_funcs:
                os.environ['NPROC_PER_NODES'] = 4 if num_infer_workers == 4 else 3
                kwargs['use_vllm'] = True
                kwargs['tensor_parallel_size'] = tp_size
                kwargs['multi_turn_func'] = multi_turn_func
                rlhf_main(
                    RLHFArguments(
                        model='Qwen/Qwen2.5-1.5B-Instruct',
                        dataset=['MATH-lighteval#100'],
                        reward_funcs=['accuracy', 'format'],
                        **kwargs))


def test_llm_zero2():
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


def test_llm_vllm_zero2():
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


def test_mllm_zero2():
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='grpo',
            model='Qwen/Qwen2-VL-2B-Instruct',
            train_type='full',
            # dataset=['AI-MO/NuminaMath-TIR#100'],
            dataset=['modelscope/coco_2014_caption:validation#100'],
            system=SYSTEM_PROMPT,
            reward_funcs=['accuracy', 'format'],
            max_completion_length=4096,
            num_generations=2,
            deepspeed='zero2',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    test_llm_pt()
    # test_llm_zero3()
    test_llm_vllm()
    # test_llm_vllm_zero2()
    # test_mllm_zero2()
