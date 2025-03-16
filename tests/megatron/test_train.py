import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def test_sft():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            load='Qwen2-7B-Instruct-mcore',
            dataset=[
                'AI-ModelScope/alpaca-gpt4-data-zh#500', 'swift/self-cognition#500',
                'AI-ModelScope/alpaca-gpt4-data-en#500'
            ],
            tensor_model_parallel_size=2,
            train_iters=100,
            model_author='swift',
            model_name='swift-robot',
            eval_iters=5,
            finetune=True))


def test_pt():
    from swift.megatron import megatron_pt_main, MegatronTrainArguments
    megatron_pt_main(
        MegatronTrainArguments(
            load='Qwen2-7B-mcore',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500', 'AI-ModelScope/alpaca-gpt4-data-en#500'],
            tensor_model_parallel_size=2,
            train_iters=200,
            eval_iters=5,
            finetune=True))


if __name__ == '__main__':
    # test_sft()
    test_pt()
