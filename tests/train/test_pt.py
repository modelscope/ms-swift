import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift import InferArguments, PretrainArguments, infer_main, pretrain_main
    result = pretrain_main(
        PretrainArguments(
            model='Qwen/Qwen2-7B-Instruct', dataset=['swift/sharegpt:all#100'], split_dataset_ratio=0.01, **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from swift import InferArguments, PretrainArguments, infer_main, pretrain_main
    result = pretrain_main(
        PretrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_pretrain_minimal():
    from swift import PretrainArguments, pretrain_main
    result = pretrain_main(
        PretrainArguments(
            model='Qwen/Qwen2-0.5B',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#20'],
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
    assert os.path.isdir(result['last_model_checkpoint'])


if __name__ == '__main__':
    # test_llm()
    test_mllm()
    # test_pretrain_minimal()
