
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test_train():
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            megatron_model='Qwen2-7B-Instruct-megatron',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh'], load_args=True))


if __name__ == '__main__':
    test_train()
