import argparse
from collections.abc import Mapping

import json
import torch
from transformers import PreTrainedTokenizerBase


def to_list(input_ids):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.cpu().numpy().tolist()
    if isinstance(input_ids, list) and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


def load_ds(ds):
    from swift.llm import load_dataset
    train_dataset, val_dataset = load_dataset(
        ds,
        split_dataset_ratio=0.0,
        strict=False,
        num_proc=1,
        model_name=['小黄', 'Xiao Huang'],
        model_author=['魔搭', 'ModelScope'])
    return train_dataset.select(range(1))


def load_and_tokenize(ms_model_id, template):
    from swift.llm import EncodePreprocessor, get_model_tokenizer, get_template
    try:
        vl_fields = ['vl', 'video', 'minicpmv', 'llava', 'vision', 'emu', 'florence']
        model_ins, tokenizer = get_model_tokenizer(ms_model_id, load_model='mplug' in ms_model_id.lower())
        template_ins = get_template(template, tokenizer)
        if template_ins.use_model:
            model_ins, _ = get_model_tokenizer(ms_model_id, load_model=True)
            template_ins.model = model_ins
        template_ins.set_mode('train')
        if 'audio' in template_ins.__class__.__name__.lower():
            output = EncodePreprocessor(template_ins)(
                load_ds('speech_asr/speech_asr_aishell1_trainsets:validation/test'))
            input_ids = output[0].get('input_ids')
        elif any([vl in template for vl in vl_fields]):
            for row in load_ds('modelscope/coco_2014_caption:validation'):
                output = template_ins.encode(row)
                input_ids = output.get('input_ids')
                # output = EncodePreprocessor(template_ins)(load_ds('swift/OK-VQA_train'))
                if model_ins is not None and model_ins.model_meta.is_multimodal:
                    inputs = template_ins.pre_data_collator([output], model=model_ins)
                    _, output = template_ins.pre_forward_hook(model_ins, None, inputs)
        else:
            output = EncodePreprocessor(template_ins)(load_ds('modelscope/DuReader_robust-QG'))
            input_ids = output[0].get('input_ids')
        if isinstance(output, Mapping):
            assert output.get('input_ids') is not None or output.get('inputs_embeds') is not None
        else:
            assert output[0].get('input_ids') is not None or output[0].get('inputs_embeds') is not None
        input_ids = to_list(input_ids)
        sent = ''
        try:
            if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
                tokenizer = tokenizer.tokenizer
            sent = tokenizer.decode(input_ids)
        except Exception:
            pass
        return input_ids, sent
    except Exception:
        import traceback
        print(traceback.format_exc())
        raise


def load_ds_old(ds):
    from swift.llm import load_dataset
    train_dataset, val_dataset = load_dataset(ds, split_dataset_ratio=0.0)
    return train_dataset.select(range(1))


def load_and_tokenize_old(ms_model_id, template):
    model_type = None
    model_info = None
    from swift.llm import get_model_tokenizer
    from swift.llm import get_template, MODEL_MAPPING
    found = False
    for model_type, model_info in MODEL_MAPPING.items():
        if model_info['model_id_or_path'].lower() == ms_model_id.lower():
            found = True
            break

    if not found:
        raise ValueError(f'No model_type found: {ms_model_id}')

    vl_fields = ['vl', 'video', 'minicpm-v', 'llava', 'vision', 'emu', 'florence']
    model_ins, tokenizer = get_model_tokenizer(model_type, load_model=True)

    if model_info['template'] == 'default-generation':
        model_info['template'] = template.replace('_', '-')
    template_ins = get_template(model_info['template'], tokenizer)
    template_ins.model = model_ins
    if 'audio' in model_info['template']:
        output = template_ins.encode(load_ds_old('aishell1-zh-mini')[0])
    elif any([vl in model_info['template'] for vl in vl_fields]):
        output = template_ins.encode(load_ds_old('coco-en-mini')[0])
    else:
        output = template_ins.encode(load_ds_old('dureader-robust-zh')[0])
    input_ids = to_list(output[0]['input_ids'])
    sent = ''
    try:
        sent = tokenizer.decode(input_ids)
    except Exception:
        pass
    return input_ids, sent


if __name__ == '__main__':
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
    parser.add_argument('--new', type=str, required=False, default='1')
    args = parser.parse_args()

    is_new = args.new == '1'
    if is_new:
        input_ids, sent = load_and_tokenize(args.ms_model_id, args.template)
    else:
        input_ids, sent = load_and_tokenize_old(args.ms_model_id, args.template)
    file = 'new_input_ids.txt' if is_new else 'old_input_ids.txt'
    if input_ids is not None:
        with open(file, 'w') as f:
            json.dump({'input_ids': input_ids, 'sent': sent}, f)
