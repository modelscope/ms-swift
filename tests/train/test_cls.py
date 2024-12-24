import os

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 10,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift.llm import TrainArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            num_labels=2,
            dataset=['simpleai/HC3-Chinese:baike_cls#200'],
            use_chat_template=False,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_mllm():
    os.environ['VIDEO_MAX_PIXELS'] = '1003520'
    os.environ['NFRAMES'] = '24'
    from swift.llm import TrainArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['AI-ModelScope/egoschema:cls#200'],
            use_chat_template=False,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    # test_llm()
    test_mllm()
