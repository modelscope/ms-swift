import argparse
from dataclasses import fields

import torch

from swift.llm import MODEL_ARCH_MAPPING, ModelKeys, get_model_tokenizer


def get_model_and_tokenizer(ms_model_id, model_arch=None):
    try:
        import transformers
        print(f'Test model: {ms_model_id} with transformers version: {transformers.__version__}')
        model_ins, tokenizer = get_model_tokenizer(ms_model_id)
        model_ins: torch.nn.Module
        if model_arch:
            model_arch: ModelKeys = MODEL_ARCH_MAPPING[model_arch]
            for f in fields(model_arch):
                value = getattr(model_arch, f.name)
                if value is not None and f.name != 'arch_name':
                    if isinstance(value, str):
                        value = [value]
                    for v in value:
                        v = v.replace('{}', '0')
                        model_ins.get_submodule(v)
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ms_model_id',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        required=True,
    )
    args = parser.parse_args()

    get_model_and_tokenizer(args.ms_model_id, args.model_arch)
