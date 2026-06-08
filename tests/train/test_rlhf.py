import os

from swift.utils import select_device

select_device('1')

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    os.environ['MAX_PIXLES'] = f'{1280 * 28 * 28}'
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['swift/RLAIF-V-Dataset#100'],
            split_dataset_ratio=0.01,
            dataset_num_proc=8,
            max_pixels=512 * 512,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_zero3():
    select_device('0,1')
    os.environ['MAX_PIXLES'] = f'{1280 * 28 * 28}'
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['swift/RLAIF-V-Dataset#100'],
            split_dataset_ratio=0.01,
            dataset_num_proc=8,
            max_pixels=512 * 512,
            deepspeed='zero3',
            **kwargs))


def test_dpo_minimal():
    from swift import InferArguments, RLHFArguments, infer_main, rlhf_main
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/orpo-dpo-mix-40k#20'],
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            save_steps=2,
            split_dataset_ratio=0.01,
            tuner_type='lora',
            logging_steps=1,
            **{
                k: v
                for k, v in kwargs.items() if k not in
                ['per_device_train_batch_size', 'save_steps', 'gradient_accumulation_steps', 'num_train_epochs']
            }))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    # test_llm()
    test_mllm()
    # test_mllm_zero3()
    # test_dpo_minimal()
