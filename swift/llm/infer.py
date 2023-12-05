# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import shutil
from typing import Tuple

import json
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm
from transformers import PreTrainedModel

from swift.tuners import Swift
from swift.utils import (append_to_jsonl, get_logger, print_model_info,
                         read_multi_line, seed_everything, show_layers)
from .utils import (InferArguments, Template, get_dataset, get_model_tokenizer,
                    get_template, inference, inference_stream)

logger = get_logger()


def merge_lora(args: InferArguments,
               replace_if_exists=False,
               device_map: str = 'cpu') -> str:
    logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.ckpt_dir is not None
    assert args.sft_type == 'lora'
    assert 'int4' not in args.model_type, 'int4 model is not supported'
    assert 'int8' not in args.model_type, 'int8 model is not supported'
    if args.quantization_bit != 0:
        logger.warning('It is not recommended to merge quantized models, '
                       'as this can result in performance degradation')
    # Loading Model and Tokenizer
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, device_map=device_map)
    logger.info(f'model_config: {model.config}')

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
            with open(old_configuration_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res.pop('adapter_cfg', None)
            with open(new_configuration_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
        # sft_args.json
        sft_args_fname = 'sft_args.json'
        old_sft_args_path = os.path.join(old_ckpt_dir, sft_args_fname)
        new_sft_args_path = os.path.join(args.ckpt_dir, sft_args_fname)
        if os.path.exists(old_sft_args_path):
            with open(old_sft_args_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res['sft_type'] = 'full'
            with open(new_sft_args_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        logger.info(f'Successfully merged LoRA and saved in {args.ckpt_dir}.')
    else:
        logger.info(
            f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
            'skipping the saving process. '
            'you can pass `replace_if_exists=True` to overwrite it.')
    return merged_lora_path


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
    kwargs = {}
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        kwargs['model_dir'] = args.ckpt_dir
    elif args.model_cache_dir is not None:
        kwargs['model_dir'] = args.model_cache_dir

    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)
    logger.info(f'model_config: {model.config}')

    # Preparing LoRA
    if args.sft_type == 'lora' and args.ckpt_dir is not None:
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    print_model_info(model)
    show_layers(model)

    template: Template = get_template(args.template_type, tokenizer,
                                      args.system, args.max_length,
                                      args.truncation_strategy)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
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
    return model, template


def llm_infer(args: InferArguments) -> None:
    if args.merge_lora_and_save:
        merge_lora(args)
    model, template = prepare_model_template(args)
    if args.overwrite_generation_config:
        assert args.ckpt_dir is not None
        model.generation_config.save_pretrained(args.ckpt_dir)
    # Inference
    result = []
    jsonl_path = None
    if args.save_result and args.ckpt_dir is not None:
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        jsonl_path = os.path.join(args.ckpt_dir, f'infer_result_{time}.jsonl')
    input_mode: Literal['S', 'M'] = 'S'
    if args.eval_human:
        logger.info('Input `exit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        if template.support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info(
                'The current template only supports single-round dialogues.')
        history = []
        while True:
            if input_mode == 'S':
                query = input('<<< ')
            else:
                query = read_multi_line()
            if query.strip().lower() == 'exit':
                break
            elif query.strip().lower() == 'clear':
                history = []
                continue
            if input_mode == 'S' and query.strip().lower() == 'multi-line':
                input_mode = 'M'
                logger.info('End multi-line input with `#`.')
                logger.info(
                    'Input `single-line` to switch to single-line input mode.')
                continue
            if input_mode == 'M' and query.strip().lower() == 'single-line':
                input_mode == 'S'
                continue
            if not template.support_multi_round:
                history = []
            gen = inference_stream(model, template, query, history)
            print_idx = 0
            for response, history in gen:
                if len(response) > print_idx:
                    print(response[print_idx:], end='', flush=True)
                    print_idx = len(response)
            print()
            print('-' * 50)
            item = history[-1]
            obj = {
                'query': item[0],
                'response': item[1],
                'history': history,
            }
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            result.append(obj)
    else:
        _, val_dataset = get_dataset(args.dataset, args.dataset_test_ratio,
                                     args.dataset_seed)
        if args.val_dataset_sample >= 0:
            val_dataset = val_dataset.select(
                range(min(args.val_dataset_sample, val_dataset.shape[0])))
        logger.info(f'val_dataset: {val_dataset}')
        if args.verbose is None:
            if len(val_dataset) >= 100:
                args.verbose = False
            else:
                args.verbose = True
            logger.info(f'Setting args.verbose: {args.verbose}')
        if not args.verbose:
            val_dataset = tqdm(val_dataset)
        for data in val_dataset:
            _, history = inference(
                model,
                template,
                data.get('query'),
                data.get('history'),
                data.get('system'),
                stream=args.stream and args.verbose,
                verbose=args.verbose)
            label = data.get('response')
            item = history[0]
            obj = {'query': item[0], 'response': item[1], 'label': label}
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            result.append(obj)
            if args.verbose:
                print()
                print(f'[LABELS]{label}')
                print('-' * 50)
    if args.save_result and args.ckpt_dir is not None:
        logger.info(f'save_result_path: {jsonl_path}')
    return {'result': result}
