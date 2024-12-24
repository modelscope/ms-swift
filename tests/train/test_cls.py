
from swift.llm import sft_main, TrainArguments


if __name__ == '__main__':
    sft_main(TrainArguments(model='Qwen/Qwen2.5-7B-Instruct',
             num_labels=2,
             dataset='simpleai/HC3-Chinese:baike_cls#1000'))
