# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import json
import torch
from transformers import GenerationConfig

from swift.tuners import Swift
from swift.tuners.rome import RomeConfig
from swift.utils import get_logger, get_main, get_model_info, seed_everything, show_layers
from .utils import (RomeArguments, Template, get_dataset, get_model_tokenizer, get_template, inference,
                    set_generation_config)

logger = get_logger()


def rome_infer(args: RomeArguments) -> None:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    logger.info('Rome does not support quantization for now, all quantization args will be ignored.')
    logger.info(f'device_count: {torch.cuda.device_count()}')

    # Loading Model and Tokenizer
    model_kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    kwargs = {'use_flash_attn': args.use_flash_attn}
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype, model_kwargs, **kwargs)
    logger.info(f'model_config: {model.config}')
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    set_generation_config(model, generation_config)
    logger.info(f'model.generation_config: {model.generation_config}')
    if args.overwrite_generation_config:
        generation_config.save_pretrained(args.ckpt_dir)

    with open(args.rome_request_file, 'r', encoding='utf-8') as f:
        request = json.load(f)

    rome_type: Optional[str] = None
    if args.model_type in ('llama2-13b-chat', 'llama2-13b', 'llama-13b-chat', 'llama-13b'):
        rome_type = 'llama-13b'
        batch_first = True
    elif args.model_type in ('llama2-7b-chat', 'llama2-7b', 'llama-7b-chat', 'llama-7b'):
        rome_type = 'llama-7b'
        batch_first = True
    elif 'chatglm' in args.model_type and '6b' in args.model_type:
        rome_type = 'chatglm-6b'
        batch_first = False

    config = RomeConfig(
        model_type=rome_type,
        knowledge=request,
        tokenizer=tokenizer,
        batch_first=batch_first,
    )
    model = Swift.prepare_model(model, config, inference_mode=True)

    show_layers(model)
    logger.info(get_model_info(model))

    # Inference
    template: Template = get_template(args.template_type, tokenizer, args.system, args.max_length,
                                      args.truncation_strategy)
    logger.info(f'system: {template.default_system}')

    # Inference
    if args.eval_human:
        while True:
            query = input('<<< ')
            inference(model, template, query, stream=args.stream, verbose=True)
    else:
        _, val_dataset = get_dataset(args.dataset, args.dataset_test_ratio, args.dataset_seed)
        mini_val_dataset = val_dataset.select(range(min(args.val_dataset_sample, val_dataset.shape[0])))
        for data in mini_val_dataset:
            inference(
                model,
                template,
                data.get('query'),
                data.get('history'),
                data.get('system'),
                stream=args.stream,
                verbose=True)
            print()
            print(f"[LABELS]{data.get('response')}")
            print('-' * 80)
            # input('next[ENTER]')


rome_main = get_main(RomeArguments, rome_infer)
