import os

from swift import SftArguments, sft_main

os.environ['MAX_PIXELS'] = str(16 * 28 * 28)

if __name__ == '__main__':
    sft_main(
        SftArguments(model='Qwen/Qwen2.5-VL-7B-Instruct', dataset='AI-ModelScope/coco#2000', split_dataset_ratio=0.01))
