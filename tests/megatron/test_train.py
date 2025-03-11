import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


def test_train():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            megatron_model='Qwen2-7B-Instruct-megatron',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500'],
            tensor_model_parallel_size=2,
            train_iters=200,
            eval_iters=5))


if __name__ == '__main__':
    test_train()
