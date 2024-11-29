import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            # ddp_find_unused_parameters=False,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            **kwargs))


def test_mllm_mp():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='full',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_llm_streaming():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-7B-Instruct', dataset=['swift/chinese-c4'], streaming=True, max_steps=16, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_mllm_streaming():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_mllm_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],  #
            deepspeed='zero3',
            **kwargs))


def test_llm_gptq():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True))


def test_llm_awq():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct-AWQ',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True))


def test_mllm_streaming_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            # deepspeed='zero3',
            **kwargs))


def test_mllm_streaming_mp_ddp():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation', 'AI-ModelScope/alpaca-gpt4-data-en'],
            streaming=True,
            max_steps=16,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            **kwargs))


def test_llm_bnb():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            quant_method='bnb',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True))


def test_moe():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True))


def test_resume_from_checkpoint():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            max_steps=5,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        TrainArguments(resume_from_checkpoint=last_model_checkpoint, load_dataset_config=True, max_steps=10))


def test_resume_only_model():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#10', 'AI-ModelScope/alpaca-gpt4-data-en#10'],
            max_steps=20,
            save_only_model=True,
            deepspeed='zero3',
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    result = sft_main(
        TrainArguments(
            resume_from_checkpoint=last_model_checkpoint,
            load_dataset_config=True,
            max_steps=20,
            resume_only_model=True))


def test_llm_transformers_4_33():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen-7B-Chat',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            **kwargs))


if __name__ == '__main__':
    # test_llm_ddp()
    # test_mllm_mp()
    # test_llm_streaming()
    # test_mllm_streaming()
    # test_mllm_zero3()
    # test_llm_gptq()
    # test_llm_awq()
    # test_mllm_streaming_zero3()
    # test_mllm_streaming_mp_ddp()
    # test_llm_bnb()
    # test_moe()
    # test_resume_from_checkpoint()
    # test_resume_only_model()
    test_llm_transformers_4_33()
