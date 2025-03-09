import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_train():
    from swift.megatron import megatron_sft_main, MegatronTrainArguments
    megatron_sft_main(
        MegatronTrainArguments(
            megatron_model='Qwen2-7B-Instruct-megatron', dataset=['AI-ModelScope/alpaca-gpt4-data-zh']))


if __name__ == '__main__':
    test_train()
