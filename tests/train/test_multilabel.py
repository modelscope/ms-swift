import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_reg_llm():
    from swift.llm import TrainArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-1.5B-Instruct',
            train_type='lora',
            num_labels=1,
            dataset=['sentence-transformers/stsb:reg#200'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, metric='acc'))


def test_reg_mllm():
    from swift.llm import TrainArguments, sft_main, infer_main, InferArguments
    # OpenGVLab/InternVL2-1B
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-2B-Instruct',
            train_type='lora',
            num_labels=1,
            dataset=['sentence-transformers/stsb:reg#200'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, metric='acc'))


if __name__ == '__main__':
    # test_reg_llm()
    test_reg_mllm()
