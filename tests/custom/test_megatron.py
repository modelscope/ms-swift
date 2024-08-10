import os

from swift.llm import ExportArguments, InferArguments, SftArguments, export_main, infer_main, sft_main

model_type = 'qwen1half-7b-chat'
tp = 2


def convert2megatron():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    export_main(ExportArguments(model_type=model_type, to_megatron=True, tp=tp, dtype='bf16'))


def convert2hf():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    export_main(ExportArguments(ckpt_dir=f'{model_type}-tp{tp}-pp1', to_hf=True, dtype='bf16'))


def sft():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    sft_main(
        SftArguments(resume_from_checkpoint=f'{model_type}-tp{tp}-pp1', dataset='alpaca-zh', train_backend='megatron'))


def infer():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    infer_main(InferArguments(model_type=model_type, model_id_or_path=f'{model_type}-tp{tp}-pp1/{model_type}-hf'))


if __name__ == '__main__':
    convert2megatron()
    # convert2hf()
    # infer()
    # sft()
