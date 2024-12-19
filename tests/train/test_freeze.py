import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_full_vit():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='full',
            freeze_llm=True,
            freeze_vit=False,
            freeze_aligner=True,
            **kwargs))


def test_full_aligner():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='full',
            freeze_llm=True,
            freeze_vit=True,
            freeze_aligner=False,
            **kwargs))


def test_lora_vit():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='lora',
            freeze_llm=True,
            freeze_vit=False,
            freeze_aligner=True,
            **kwargs))


def test_lora_aligner():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='lora',
            freeze_llm=True,
            freeze_vit=True,
            freeze_aligner=False,
            **kwargs))


if __name__ == '__main__':
    # test_full_vit()
    test_full_aligner()
    # test_lora_vit()
    # test_lora_aligner()
