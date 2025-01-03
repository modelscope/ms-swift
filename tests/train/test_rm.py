import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_infer():
    from swift.llm import infer_main, InferArguments
    infer_main(
        InferArguments(
            model='Shanghai_AI_Laboratory/internlm2-1_8b-reward', val_dataset='AI-ModelScope/alpaca-gpt4-data-zh#500'))


def test_llm():
    from swift.llm import TrainArguments, sft_main, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Shanghai_AI_Laboratory/internlm2-1_8b-reward',
            train_type='lora',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji:zh#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    # test_infer()
    test_llm()
