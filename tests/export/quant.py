import os
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_llm_quant(quant_method: Literal['gptq', 'awq'] = 'awq'):
    from swift.llm import export_main, ExportArguments
    export_main(
        ExportArguments(
            model='qwen/Qwen2-7B-Instruct',
            quant_bits=4,
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#1000'],
            quant_method=quant_method))


def test_vlm_quant(quant_method: Literal['gptq', 'awq'] = 'awq'):
    from swift.llm import export_main, ExportArguments
    export_main(
        ExportArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            quant_bits=4,
            dataset=['modelscope/coco_2014_caption:val#1000'],
            quant_method=quant_method))


if __name__ == '__main__':
    # test_llm_quant('gptq')
    test_vlm_quant('gptq')
