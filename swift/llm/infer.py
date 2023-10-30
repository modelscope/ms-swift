# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Tuple

import json
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from transformers import PreTrainedModel

from swift.tuners import Swift
from swift.utils import (get_logger, print_model_info, seed_everything,
                         show_layers)
from .utils import (InferArguments, Template, get_dataset, get_model_tokenizer,
                    get_template, inference)

logger = get_logger()


def merge_lora(args: InferArguments, replace_if_exists=False) -> None:
    assert args.ckpt_dir is not None
    assert args.sft_type == 'lora'
    assert not args.model_type.endswith('int4'), 'int4 model is not supported'
    assert not args.model_type.endswith('int8'), 'int8 model is not supported'
    if args.quantization_bit != 0:
        logger.warning('It is not recommended to merge quantized models, '
                       'as this can result in performance degradation')
    # Loading Model and Tokenizer
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, device_map='cpu')

    # Preparing LoRA
    model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
    Swift.merge_and_unload(model)
    model = model.model

    old_ckpt_dir = args.ckpt_dir
    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path

    if not os.path.exists(args.ckpt_dir) or replace_if_exists:
        logger.info('Saving merged weights...')
        model.save_pretrained(args.ckpt_dir)
        tokenizer.save_pretrained(args.ckpt_dir)
        for fname in os.listdir(old_ckpt_dir):
            if fname in {'generation_config.json'}:
                src_path = os.path.join(old_ckpt_dir, fname)
                tgt_path = os.path.join(args.ckpt_dir, fname)
                shutil.copy(src_path, tgt_path)
        # configuration.json
        configuration_fname = 'configuration.json'
        old_configuration_path = os.path.join(old_ckpt_dir,
                                              configuration_fname)
        new_configuration_path = os.path.join(args.ckpt_dir,
                                              configuration_fname)
        if os.path.exists(old_configuration_path):
            with open(old_configuration_path, 'r') as f:
                res = json.load(f)
            res.pop('adapter_cfg', None)
            with open(new_configuration_path, 'w') as f:
                json.dump(res, f)
        logger.info('Successfully merged LoRA.')
    else:
        logger.info('The weight directory for the merged LoRA already exists, '
                    'skipping the saving process.')


def prepare_model_template(
        args: InferArguments) -> Tuple[PreTrainedModel, Template]:
    logger.info(f'args: {args}')
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # Loading Model and Tokenizer
    model_kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config
    kwargs = {'use_flash_attn': args.use_flash_attn}
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        kwargs['model_dir'] = args.ckpt_dir
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)

    # Preparing LoRA
    if args.sft_type == 'lora' and args.ckpt_dir is not None:
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    print_model_info(model)
    show_layers(model)

    template: Template = get_template(args.template_type, tokenizer,
                                      args.system, args.max_length)
    generation_config = GenerationConfig(
        max_length=None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')
    model.generation_config = generation_config
    return model, template


def llm_infer(args: InferArguments) -> None:
    if args.merge_lora_and_save:
        merge_lora(args)
    model, template = prepare_model_template(args)
    if args.overwrite_generation_config:
        assert args.ckpt_dir is not None
        model.generation_config.save_pretrained(args.ckpt_dir)
    # Inference
    if args.eval_human:
        while True:
            query = input('<<< ')
            inference(model, template, query, stream=args.stream)
    else:
        _, val_dataset = get_dataset(args.dataset, args.dataset_test_ratio,
                                     args.dataset_seed)
        mini_val_dataset = val_dataset.select(
            range(min(args.show_dataset_sample, val_dataset.shape[0])))
        for data in mini_val_dataset:
            inference(
                model,
                template,
                data.get('query'),
                data.get('history'),
                data.get('system'),
                stream=args.stream)
            print()
            print(f"[LABELS]{data.get('response')}")
            print('-' * 80)
            # input('next[ENTER]')
