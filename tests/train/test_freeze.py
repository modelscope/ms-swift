import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
    'metric_for_best_model': 'loss'
}


def test_freeze_vit_full():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='full',
            **kwargs))


def test_freeze_llm():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='lora',
            **kwargs))


def test_freeze_aligner():
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['SIZE_FACTOR'] = '12'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    sft_main(
        TrainArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            train_type='lora',
            **kwargs))


if __name__ == '__main__':
    test_freeze_vit_full()
    # test_freeze_llm()
    # test_freeze_aligner()
