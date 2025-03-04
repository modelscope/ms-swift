import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def hf2megatron():
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model='Qwen/Qwen2.5-7B-Instruct', to_megatron=True, torch_dtype='bfloat16'))


def megatron2hf():
    from swift.llm import export_main, ExportArguments


if __name__ == '__main__':
    hf2megatron()
