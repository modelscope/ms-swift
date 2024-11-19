import argparse
from dataclasses import fields

import torch

from swift.llm import (MODEL_ARCH_MAPPING, MODEL_MAPPING, EncodePreprocessor, ModelKeys, get_model_tokenizer,
                       get_template, load_dataset)


def load_ds(ds):
    train_dataset, val_dataset = load_dataset(
        ds,
        split_dataset_ratio=0.0,
        strict=False,
        num_proc=1,
        model_name=['小黄', 'Xiao Huang'],
        model_author=['魔搭', 'ModelScope'])
    return train_dataset.select(range(min(10, len(train_dataset))))


def load_and_tokenize(ms_model_id, template):
    try:
        load_model = False
        if 'gen' in template or 'audio' in template or 'vl' in template:
            load_model = True
        model_ins, tokenizer = get_model_tokenizer(ms_model_id, load_model=load_model)
        template_ins = get_template(template, tokenizer)
        template_ins.set_mode('train')
        if 'audio' in template_ins.__class__.__name__.lower():
            output = EncodePreprocessor(
                template_ins, model=model_ins)(
                    load_ds('speech_asr/speech_asr_aishell1_trainsets:validation'))
        elif 'vl' in template_ins.__class__.__name__.lower() or 'gen' in template_ins.__class__.__name__.lower(
        ) or 'video' in template_ins.__class__.__name__.lower():
            for row in load_ds('swift/OK-VQA_train'):
                output = template_ins.encode(row)
                # output = EncodePreprocessor(template_ins)(load_ds('swift/OK-VQA_train'))
                if model_ins is not None and model_ins.model_meta.is_multimodal:
                    inputs = template_ins.pre_data_collator([output], padding_side='left', model=model_ins)
                    _, output = template_ins.pre_forward_hook(model_ins, None, inputs, padding_side='left')
        else:
            output = EncodePreprocessor(template_ins)(load_ds('AI-ModelScope/sharegpt_gpt4:default'))
        if isinstance(output, dict):
            assert output.get('input_ids') is not None or output.get('inputs_embeds') is not None
        else:
            assert output[0].get('input_ids') is not None or output[0].get('inputs_embeds') is not None
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise


parser = argparse.ArgumentParser()
parser.add_argument(
    '--ms_model_id',
    type=str,
    required=True,
)
parser.add_argument(
    '--template',
    type=str,
    required=True,
)
args = parser.parse_args()

load_and_tokenize(args.ms_model_id, args.template)
