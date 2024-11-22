import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'metric_for_best_model': 'loss'
}


def test_llm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_mllm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_llm_streaming():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100', 'AI-ModelScope/alpaca-gpt4-data-en#100'],
            streaming=True,
            max_steps=16,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    # test_mllm()
    test_llm_streaming()
