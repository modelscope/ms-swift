# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import shutil
from typing import Any, Dict, Literal, Optional, Tuple

import json
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm
from transformers import PreTrainedModel

from swift.tuners import Swift
from swift.utils import (append_to_jsonl, get_logger, get_main, get_model_info,
                         read_multi_line, seed_everything, show_layers)
from .utils import (TEMPLATE_MAPPING, InferArguments, Template,
                    get_additional_saved_files, get_dataset,
                    get_model_tokenizer, get_template, inference,
                    inference_stream, is_lora, set_generation_config)

logger = get_logger()


def merge_lora(args: InferArguments,
               replace_if_exists=False,
               device_map: str = 'auto',
               **kwargs) -> Optional[str]:
    logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
    assert args.sft_type == 'lora', "Only supports sft_type == 'lora'"
    assert 'int4' not in args.model_type, 'int4 model is not supported'
    assert 'int8' not in args.model_type, 'int8 model is not supported'
    if args.quantization_bit != 0:
        logger.warning('It is not recommended to merge quantized models, '
                       'as this can result in performance degradation')
    old_ckpt_dir = args.ckpt_dir
    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path
    if os.path.exists(args.ckpt_dir) and not replace_if_exists:
        logger.info(
            f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
            'skipping the saving process. '
            'you can pass `replace_if_exists=True` to overwrite it.')
        return
    # Loading Model and Tokenizer
    kwargs = {}
    model_kwargs = {'low_cpu_mem_usage': True, 'device_map': device_map}
    if args.model_cache_dir is not None:
        kwargs['model_dir'] = args.model_cache_dir
    model, tokenizer = get_model_tokenizer(args.model_type, args.torch_dtype,
                                           model_kwargs, **kwargs)
    logger.info(f'model_config: {model.config}')

    # Preparing LoRA
    model = Swift.from_pretrained(model, old_ckpt_dir, inference_mode=True)
    Swift.merge_and_unload(model)
    model = model.model
    logger.info('Saving merged weights...')
    model.save_pretrained(
        merged_lora_path, safe_serialization=args.save_safetensors)
    for add_file in get_additional_saved_files(args.model_type):
        shutil.copy(
            os.path.join(model.model_dir, add_file),
            os.path.join(merged_lora_path, add_file))
    tokenizer.save_pretrained(merged_lora_path)
    for fname in os.listdir(old_ckpt_dir):
        if fname in {'generation_config.json'}:
            src_path = os.path.join(old_ckpt_dir, fname)
            tgt_path = os.path.join(merged_lora_path, fname)
            shutil.copy(src_path, tgt_path)
    # configuration.json
    configuration_fname = 'configuration.json'
    old_configuration_path = os.path.join(old_ckpt_dir, configuration_fname)
    new_configuration_path = os.path.join(merged_lora_path,
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
    new_sft_args_path = os.path.join(merged_lora_path, sft_args_fname)
    if os.path.exists(old_sft_args_path):
        with open(old_sft_args_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        res['sft_type'] = 'full'
        with open(new_sft_args_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    logger.info(f'Successfully merged LoRA and saved in {merged_lora_path}.')
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
    logger.info(f'generation_config: {generation_config}')
    set_generation_config(model, generation_config)
    # Preparing LoRA
    if is_lora(args.sft_type) and args.ckpt_dir is not None:
        model = Swift.from_pretrained(
            model, args.ckpt_dir, inference_mode=True)

    logger.info(get_model_info(model))
    show_layers(model)

    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    return model, template


def read_media_file(
        infer_kwargs: Dict[str, Any],
        infer_media_type: Literal['none', 'round', 'dialogue']) -> None:
    text = 'Input a media path or URL <<< '
    images = infer_kwargs.get('images', [])
    if infer_media_type == 'none':
        return
    if infer_media_type == 'round' or len(images) == 0:
        image = input(text)
        if len(image) > 0:
            images += [image]
    if len(images) > 0:
        infer_kwargs['images'] = images


def llm_infer(args: InferArguments) -> None:
    if args.merge_lora_and_save:
        merge_lora(args, device_map='cpu')
    if args.infer_backend == 'vllm':
        from swift.llm import prepare_vllm_engine_template, inference_stream_vllm, inference_vllm
        llm_engine, template = prepare_vllm_engine_template(args)
    else:
        model, template = prepare_model_template(args)
        if args.overwrite_generation_config:
            assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
            model.generation_config.save_pretrained(args.ckpt_dir)
    # Inference
    result = []
    jsonl_path = None
    if args.save_result and args.ckpt_dir is not None:
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        jsonl_path = os.path.join(args.ckpt_dir, f'infer_result_{time}.jsonl')
    if args.eval_human:
        input_mode: Literal['S', 'M'] = 'S'
        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        if template.support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info(
                'The current template only supports single-round dialogues.')
        history = []
        infer_kwargs = {}
        if args.infer_media_type != 'none':
            logger.info('Please enter the conversation content first, '
                        'followed by the path to the multimedia file.')
        while True:
            if input_mode == 'S':
                query = input('<<< ')
            else:
                query = read_multi_line()
            if query.strip().lower() in {'exit', 'quit'}:
                break
            elif query.strip().lower() == 'clear':
                history = []
                infer_kwargs = {}
                continue
            elif query.strip() == '':
                continue
            if input_mode == 'S' and query.strip().lower() == 'multi-line':
                input_mode = 'M'
                logger.info('End multi-line input with `#`.')
                logger.info(
                    'Input `single-line` to switch to single-line input mode.')
                continue
            if input_mode == 'M' and query.strip().lower() == 'single-line':
                input_mode = 'S'
                continue
            if not template.support_multi_round:
                history = []
                infer_kwargs = {}
            read_media_file(infer_kwargs, args.infer_media_type)
            if args.infer_backend == 'vllm':
                request_list = [{'query': query, 'history': history}]
                if args.stream:
                    gen = inference_stream_vllm(llm_engine, template,
                                                request_list)
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        new_history = resp_list[0]['history']
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    resp_list = inference_vllm(llm_engine, template,
                                               request_list)
                    response = resp_list[0]['response']
                    new_history = resp_list[0]['history']
                    print(response)
            else:
                if args.stream:
                    gen = inference_stream(model, template, query, history,
                                           **infer_kwargs)
                    print_idx = 0
                    for response, new_history in gen:
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    response, new_history = inference(model, template, query,
                                                      history, **infer_kwargs)
                    print(response)
            print('-' * 50)
            obj = {
                'query': query,
                'response': response,
                'history': history,
            }
            history = new_history
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
        if not args.verbose and args.stream:
            args.stream = False
            logger.info(f'Setting args.stream: {args.stream}')

        if args.infer_backend == 'vllm' and not args.stream:
            if args.verbose:
                args.verbose = False
                logger.info('Setting args.verbose: False')
            label_list = None
            if 'response' in val_dataset.features:
                label_list = val_dataset['response']
            val_dataset = val_dataset.remove_columns('response')
            request_list = val_dataset.to_list()
            resp_list = inference_vllm(
                llm_engine, template, request_list, use_tqdm=True)
            result = []
            if label_list is not None:
                for request, label in zip(request_list, label_list):
                    request['label'] = label
            for request, resp in zip(request_list, resp_list):
                obj = {'response': resp['response'], **request}
                if jsonl_path is not None:
                    append_to_jsonl(jsonl_path, obj)
                result.append(obj)
        else:
            if not args.verbose:
                val_dataset = tqdm(val_dataset)
            for data in val_dataset:
                kwargs = {'query': data['query']}
                history = data.get('history')
                system = data.get('system')
                images = data.get('images')
                if history is not None:
                    kwargs['history'] = history
                if system is not None:
                    kwargs['system'] = system
                if images is not None:
                    kwargs['images'] = images
                if args.infer_backend == 'vllm':
                    assert args.stream is True
                    if args.verbose:
                        print(f"query: {data['query']}\nresponse: ", end='')
                    gen = inference_stream_vllm(llm_engine, template, [kwargs])
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        if args.verbose and len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    response, _ = inference(
                        model,
                        template,
                        stream=args.stream and args.verbose,
                        verbose=args.verbose,
                        **kwargs)
                label = data.pop('response')
                if label is not None:
                    kwargs['label'] = label
                obj = {'response': response, **kwargs}
                if jsonl_path is not None:
                    append_to_jsonl(jsonl_path, obj)
                result.append(obj)
                if args.verbose:
                    print()
                    print(f'[LABELS]{label}')
                    if images is not None:
                        print(f'[IMAGES]{images}')
                    print('-' * 50)
    if args.save_result and args.ckpt_dir is not None:
        logger.info(f'save_result_path: {jsonl_path}')
    return {'result': result}


infer_main = get_main(InferArguments, llm_infer)
merge_lora_main = get_main(InferArguments, merge_lora)
