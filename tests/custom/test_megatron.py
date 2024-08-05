import os

from swift.llm import ExportArguments, InferArguments, SftArguments, export_main, infer_main, sft_main


def convert2megatron():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    export_main(ExportArguments(model_type='qwen2-7b-instruct', to_megatron=True, tp=2))


def convert2hf():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    export_main(ExportArguments(ckpt_dir='qwen2-7b-instruct-tp2-pp1', to_hf=True, tp=2))


def test_convert():
    convert2megatron()
    convert2hf()


def sft():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    sft_main(
        SftArguments(resume_from_checkpoint='qwen2-7b-instruct-tp2-pp1', dataset='alpaca-zh', train_backend='megatron'))


def infer():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    infer_main(
        InferArguments(
            model_type='qwen2-7b-instruct', model_id_or_path='qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf'))


if __name__ == '__main__':
    test_convert()
    # infer()
    # sft()
