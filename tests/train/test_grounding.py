import os

from swift.llm import TrainArguments, sft_main

os.environ['MAX_PIXELS'] = str(16 * 28 * 28)

if __name__ == '__main__':
    sft_main(TrainArguments(model='Qwen/Qwen2.5-VL-7B-Instruct', dataset='AI-ModelScope/coco#2000'))
