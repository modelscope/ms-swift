# Copyright (c) Alibaba, Inc. and its affiliates.
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (InferArguments, get_dataset, get_model_tokenizer,
                   get_preprocess, inference, show_layers)

from swift import Swift, get_logger
from swift.utils import parse_args, print_model_info, seed_everything

logger = get_logger()


def llm_infer(args: InferArguments) -> None:
    args.init_argument()
    logger.info(f'args: {args}')
    if not os.path.isdir(args.ckpt_dir):
        raise ValueError(f'Please enter a valid ckpt_dir: {args.ckpt_dir}')
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config
    if args.model_type.startswith('qwen'):
        kwargs['use_flash_attn'] = args.use_flash_attn

    if args.sft_type == 'full':
        kwargs['model_dir'] = args.ckpt_dir
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing LoRA
    if args.sft_type == 'lora':
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    show_layers(model)
    print_model_info(model)

    # ### Inference
    preprocess_func = get_preprocess(args.template_type, tokenizer,
                                     args.system, args.max_length)
    streamer = None
    if args.use_streamer:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
    generation_config = GenerationConfig(
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

    if args.eval_human:
        while True:
            query = input('<<< ')
            data = {'query': query}
            input_ids = preprocess_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer)
    else:
        _, val_dataset = get_dataset(
            args.dataset.split(','), args.dataset_test_ratio,
            args.dataset_split_seed)
        mini_val_dataset = val_dataset.select(
            range(min(args.show_dataset_sample, val_dataset.shape[0])))
        for data in mini_val_dataset:
            response = data['response']
            data['response'] = None
            input_ids = preprocess_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer)
            print()
            print(f'[LABELS]{response}')
            print('-' * 80)
            # input('next[ENTER]')


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_infer(args)
