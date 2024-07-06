# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import shutil
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
import numpy as np
import torch
from modelscope import BitsAndBytesConfig, GenerationConfig
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import is_torch_npu_available

from swift.tuners import Swift
from swift.utils import (append_to_jsonl, get_logger, get_main, get_model_info, read_multi_line, seed_everything,
                         show_layers)
from .utils import (DeployArguments, InferArguments, MediaTag, Template, get_additional_saved_files, get_dataset,
                    get_model_tokenizer, get_template, inference, inference_stream, is_adapter, is_quant_model,
                    sample_dataset, set_generation_config)

logger = get_logger()


def save_checkpoint(model: Optional[PreTrainedModel],
                    tokenizer: PreTrainedTokenizerBase,
                    model_cache_dir: str,
                    ckpt_dir: Optional[str],
                    target_dir: str,
                    *,
                    save_safetensors: bool = True,
                    sft_args_kwargs: Dict[str, Any],
                    **kwargs) -> None:
    if model is not None:
        model.save_pretrained(target_dir, safe_serialization=save_safetensors)
    if hasattr(tokenizer, 'processor'):
        tokenizer.processor.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    model_type = getattr(tokenizer, 'model_type')
    fname_list = ['generation_config.json', 'preprocessor_config.json']
    if model_type is not None:
        fname_list += get_additional_saved_files(model_type)

    for fname in fname_list:
        tgt_path = os.path.join(target_dir, fname)
        for model_dir in [ckpt_dir, model_cache_dir]:
            if model_dir is None:
                continue
            src_path = os.path.join(model_dir, fname)
            if os.path.isfile(src_path):
                shutil.copy(src_path, tgt_path)
                break
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, tgt_path)
                break
    # configuration.json
    configuration_fname = 'configuration.json'
    new_configuration_path = os.path.join(target_dir, configuration_fname)
    for model_dir in [ckpt_dir, model_cache_dir]:
        if model_dir is None:
            continue
        old_configuration_path = os.path.join(model_dir, configuration_fname)
        if os.path.exists(old_configuration_path):
            with open(old_configuration_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res.pop('adapter_cfg', None)
            with open(new_configuration_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
            break
    if ckpt_dir is not None:
        # sft_args.json
        sft_args_fname = 'sft_args.json'
        old_sft_args_path = os.path.join(ckpt_dir, sft_args_fname)
        new_sft_args_path = os.path.join(target_dir, sft_args_fname)
        if os.path.exists(old_sft_args_path):
            with open(old_sft_args_path, 'r', encoding='utf-8') as f:
                res = json.load(f)
            res['sft_type'] = 'full'
            for k in ['dtype', 'quant_method']:
                v = sft_args_kwargs.get(k)
                if v is not None:
                    res[k] = v
            with open(new_sft_args_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)


def merge_lora(args: InferArguments,
               replace_if_exists=False,
               device_map: Optional[str] = None,
               **kwargs) -> Optional[str]:
    logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
    assert args.sft_type in ('lora', 'adalora', 'longlora', 'llamapro'), 'Only supports lora & llamapro series models'
    assert not is_quant_model(
        args.model_type), f'{args.model_type} is a quantized model and does not support merge-lora.'
    if args.quantization_bit != 0:
        logger.warning('It is not recommended to merge quantized models, '
                       'as this can result in performance degradation')
    ckpt_dir, ckpt_name = os.path.split(args.ckpt_dir)
    merged_lora_path = os.path.join(ckpt_dir, f'{ckpt_name}-merged')
    logger.info(f'merged_lora_path: `{merged_lora_path}`')
    if os.path.exists(merged_lora_path) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {args.ckpt_dir}, '
                    'skipping the saving process. '
                    'you can pass `replace_if_exists=True` to overwrite it.')
    else:
        if device_map is None:
            device_map = args.merge_device_map
        logger.info(f'merge_device_map: {device_map}')
        model, template = prepare_model_template(args, device_map=device_map, verbose=False)
        logger.info('Merge LoRA...')
        Swift.merge_and_unload(model)
        model = model.model
        logger.info('Saving merged weights...')
        save_checkpoint(
            model,
            template.tokenizer,
            model.model_dir,
            args.ckpt_dir,
            merged_lora_path,
            save_safetensors=args.save_safetensors,
            sft_args_kwargs={'dtype': args.dtype})
        logger.info(f'Successfully merged LoRA and saved in {merged_lora_path}.')
    logger.info("Setting args.sft_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.sft_type = 'full'
    args.ckpt_dir = merged_lora_path
    return merged_lora_path


def prepare_model_template(args: InferArguments,
                           *,
                           device_map: Optional[str] = None,
                           verbose: bool = True,
                           automodel_class=None) -> Tuple[PreTrainedModel, Template]:

    model_kwargs = {}
    if is_torch_npu_available():
        logger.info(f'device_count: {torch.npu.device_count()}')
        if device_map is None:
            device_map = 'npu:0'
    else:
        logger.info(f'device_count: {torch.cuda.device_count()}')
        if device_map is None:
            device_map = 'auto' if torch.cuda.device_count() > 1 else 'cuda:0'
    if device_map == 'auto':
        model_kwargs['low_cpu_mem_usage'] = True
    model_kwargs['device_map'] = device_map

    # Loading Model and Tokenizer
    if hasattr(args, 'quant_config'):
        model_kwargs['quantization_config'] = args.quant_config
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        if args.bnb_4bit_compute_dtype is None:
            quantization_config.bnb_4bit_compute_dtype = None
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        model_kwargs['quantization_config'] = quantization_config
    kwargs = {}
    if args.use_flash_attn is not None:
        kwargs['use_flash_attn'] = args.use_flash_attn
    model_id_or_path = None
    if args.sft_type == 'full' and args.ckpt_dir is not None:
        model_id_or_path = args.ckpt_dir
    elif args.model_id_or_path is not None:
        model_id_or_path = args.model_id_or_path
    if automodel_class is not None:
        kwargs['automodel_class'] = automodel_class
    if args.local_repo_path:
        kwargs['local_repo_path'] = args.local_repo_path
    if args.rope_scaling:
        kwargs['rope_scaling'] = args.rope_scaling
        kwargs['max_length'] = args.max_length
    model, tokenizer = get_model_tokenizer(
        args.model_type,
        args.torch_dtype,
        model_kwargs,
        model_id_or_path=model_id_or_path,
        revision=args.model_revision,
        quant_method=args.quant_method,
        **kwargs)
    if verbose:
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

    if model.max_model_len is None:
        model.max_model_len = args.max_model_len
    elif args.max_model_len is not None:
        if args.max_model_len <= model.max_model_len:
            model.max_model_len = args.max_model_len
        else:
            raise ValueError('args.max_model_len exceeds the maximum max_model_len supported by the model.'
                             f'args.max_model_len: {args.max_model_len}, model.max_model_len: {model.max_model_len}')
    # Preparing LoRA
    if is_adapter(args.sft_type) and args.ckpt_dir is not None:
        if is_quant_model(args.model_type, model):
            # gptq awq does not support lora switching
            args.lora_request_list = None
            logger.warning('The current model does not support LoRA switching. '
                           f'Setting args.lora_request_list: {args.lora_request_list}')
        if isinstance(args, DeployArguments) and args.lora_request_list is not None:
            logger.info(f'args.lora_request_list: {args.lora_request_list}')
            for lora_request in args.lora_request_list:
                model = Swift.from_pretrained(
                    model, lora_request.lora_local_path, lora_request.lora_name, inference_mode=True)
        else:
            model = Swift.from_pretrained(model, args.ckpt_dir, inference_mode=True)
        model = model.to(model.dtype)
    model.requires_grad_(False)

    if verbose:
        show_layers(model)
        logger.info(model)
    logger.info(get_model_info(model))
    template: Template = get_template(
        args.template_type,
        tokenizer,
        args.system,
        args.max_length,
        args.truncation_strategy,
        model=model,
        tools_prompt=args.tools_prompt)
    args.system = template.default_system
    logger.info(f'system: {args.system}')
    return model, template


def read_media_file(infer_kwargs: Dict[str, Any], infer_media_type: Literal['none', 'round', 'dialogue'],
                    media_type: Literal['image', 'video', 'audio']) -> None:
    media_key = MediaTag.media_keys[media_type]
    a_an = 'an' if media_type[0] in {'i', 'a'} else 'a'
    text = f'Input {a_an} {media_type} path or URL <<< '
    media_files = infer_kwargs.get(media_key) or []
    if infer_media_type == 'none':
        return
    if infer_media_type == 'round' or len(media_files) == 0:
        media_files += [input(text) or None]
    if len(media_files) > 0:
        infer_kwargs[media_key] = media_files


def llm_infer(args: InferArguments) -> Dict[str, List[Dict[str, Any]]]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    if args.merge_lora:
        merge_lora(args, device_map=args.merge_device_map)

    if args.infer_backend == 'vllm':
        from .utils import prepare_vllm_engine_template, inference_stream_vllm, inference_vllm
        llm_engine, template = prepare_vllm_engine_template(args)
    else:
        device_map = None
        if args.device_map_config_path is not None:
            cwd = os.getcwd()
            config_path = args.device_map_config_path if os.path.isabs(args.device_map_config_path) else os.path.join(
                cwd, args.device_map_config_path)
            with open(config_path, 'r') as json_file:
                device_map = json.load(json_file)
        if args.quant_method == 'hqq':
            from transformers import HqqConfig
            if args.hqq_dynamic_config_path is not None:
                cwd = os.getcwd()
                config_path = args.hqq_dynamic_config_path if os.path.isabs(
                    args.hqq_dynamic_config_path) else os.path.join(cwd, args.hqq_dynamic_config_path)
                with open(config_path, 'r') as json_file:
                    args.quant_config = HqqConfig(dynamic_config=json.load(json_file))
            else:
                if args.quantization_bit == 0:
                    logger.info("You haven't set the quantization_bit parameter; set it to 8.")
                    args.quantization_bit = 8
                args.quant_config = HqqConfig(nbits=args.quantization_bit, axis=args.hqq_axis)
        elif args.quant_method == 'eetq':
            from transformers import EetqConfig
            args.quant_config = EetqConfig('int8')
        model, template = prepare_model_template(args, device_map=device_map)
        if args.overwrite_generation_config:
            assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
            model.generation_config.save_pretrained(args.ckpt_dir)
    lora_request = None
    if args.vllm_enable_lora:
        assert len(args.lora_request_list) == 1
        lora_request = args.lora_request_list[0]
    # Inference
    result: List[Dict[str, Any]] = []
    jsonl_path = None
    if args.save_result:
        result_dir = args.ckpt_dir
        if result_dir is None:
            result_dir = llm_engine.model_dir if args.infer_backend == 'vllm' else model.model_dir
        if result_dir is not None:
            result_dir = os.path.join(result_dir, 'infer_result')
            os.makedirs(result_dir, exist_ok=True)
            time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            jsonl_path = os.path.join(result_dir, f'{time}.jsonl')
    if args.eval_human:
        input_mode: Literal['S', 'M'] = 'S'
        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        if template.support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')
        history = []
        infer_kwargs = {}
        if args.infer_media_type != 'none':
            logger.info('Please enter the conversation content first, ' 'followed by the path to the multimedia file.')
        system = None
        read_system = False
        while True:
            if input_mode == 'S':
                addi_prompt = ''
                if read_system:
                    addi_prompt = '[S]'
                query = input(f'<<<{addi_prompt} ')
            else:
                addi_prompt = '[M]'
                if read_system:
                    addi_prompt = '[MS]'
                query = read_multi_line(addi_prompt)
            if query.strip().lower() in {'exit', 'quit'}:
                break
            elif query.strip().lower() == 'clear':
                history = []
                infer_kwargs = {}
                continue
            elif query.strip() == '' and not read_system:
                continue
            elif query.strip().lower() == 'reset-system':
                read_system = True
                continue
            if read_system:
                if query == '':
                    system = None
                else:
                    system = query
                read_system = False
                history = []
                infer_kwargs = {}
                continue
            if input_mode == 'S' and query.strip().lower() == 'multi-line':
                input_mode = 'M'
                logger.info('End multi-line input with `#`.')
                logger.info('Input `single-line` to switch to single-line input mode.')
                continue
            if input_mode == 'M' and query.strip().lower() == 'single-line':
                input_mode = 'S'
                continue
            if not template.support_multi_round:
                history = []
                infer_kwargs = {}

            read_media_file(infer_kwargs, args.infer_media_type, args.media_type)
            infer_kwargs['truncation_strategy'] = args.truncation_strategy
            if system is None and template.use_default_system:
                system = template.default_system
            if args.infer_backend == 'vllm':
                request_list = [{'query': query, 'history': history, 'system': system, **infer_kwargs}]
                if args.stream:
                    gen = inference_stream_vllm(llm_engine, template, request_list, lora_request=lora_request)
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        new_history = resp_list[0]['history']
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    resp_list = inference_vllm(llm_engine, template, request_list, lora_request=lora_request)
                    response = resp_list[0]['response']
                    new_history = resp_list[0]['history']
                    print(response)
            else:
                if args.stop_words:
                    infer_kwargs['stop_words'] = args.stop_words
                if args.stream:
                    gen = inference_stream(model, template, query, history, system, **infer_kwargs)
                    print_idx = 0
                    for response, new_history in gen:
                        if len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    response, new_history = inference(model, template, query, history, system, **infer_kwargs)
                    print(response)
            print('-' * 50)
            obj = {
                'system': system,
                'query': query,
                'response': response,
                'history': history,
            }
            for media_key in MediaTag.media_keys.values():
                media_files = infer_kwargs.get(media_key)
                if media_files is not None:
                    obj[media_key] = media_files
            history = new_history
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            result.append(obj)
    else:
        dataset_kwargs = {
            'dataset_seed': args.dataset_seed,
            'check_dataset_strategy': args.check_dataset_strategy,
            'model_name': args.model_name,
            'model_author': args.model_author
        }
        if len(args.val_dataset) > 0:
            _, val_dataset = get_dataset(args.val_dataset, 1.0, **dataset_kwargs)
        else:
            _, val_dataset = get_dataset(args.dataset, args.dataset_test_ratio, **dataset_kwargs)
        _, val_dataset = args._handle_dataset_compat(_, val_dataset)
        assert val_dataset is not None
        if 0 <= args.show_dataset_sample < val_dataset.shape[0]:
            random_state = np.random.RandomState(args.dataset_seed)
            logger.info(f'show_dataset_sample: {args.show_dataset_sample}')
            val_dataset = sample_dataset(val_dataset, args.show_dataset_sample, random_state)
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
            request_list = []
            for data in val_dataset:
                request = {'query': data['query']}
                history = data.get('history')
                system = data.get('system')
                if history is None:
                    history = []
                request['history'] = history
                if system is None and template.use_default_system:
                    system = template.default_system
                request['system'] = system
                for media_key in MediaTag.media_keys.values():
                    media_files = data.get(media_key)
                    if media_files is not None:
                        request[media_key] = media_files
                request['truncation_strategy'] = args.truncation_strategy
                request_list.append(request)
            resp_list = inference_vllm(llm_engine, template, request_list, use_tqdm=True)
            result = []
            if label_list is not None:
                for request, label in zip(request_list, label_list):
                    request['label'] = label
            for request, resp in zip(request_list, resp_list):
                obj = {
                    'system': request['system'],
                    'query': request['query'],
                    'response': resp['response'],
                    'label': request.pop('label', None),
                    'history': request['history'],
                }
                for media_key in MediaTag.media_keys.values():
                    media_files = request.get(media_key)
                    if media_files is not None:
                        obj[media_key] = media_files
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
                tools = data.get('tools')
                objects = data.get('objects')
                if args.verbose and system is not None:
                    print(f'[SYSTEM]{system}')
                if history is None:
                    history = []
                kwargs['history'] = history
                if system is None and template.use_default_system:
                    system = template.default_system
                kwargs['system'] = system
                for media_key in MediaTag.media_keys.values():
                    media_files = data.get(media_key)
                    if media_files is not None:
                        kwargs[media_key] = media_files
                if tools is not None:
                    kwargs['tools'] = tools
                if objects is not None:
                    kwargs['objects'] = objects
                kwargs['truncation_strategy'] = args.truncation_strategy
                if args.infer_backend == 'vllm':
                    assert args.stream
                    if args.verbose:
                        print(f"[QUERY]{data['query']}\n[RESPONSE]", end='')
                    gen = inference_stream_vllm(llm_engine, template, [kwargs], lora_request=lora_request)
                    print_idx = 0
                    for resp_list in gen:
                        response = resp_list[0]['response']
                        if args.verbose and len(response) > print_idx:
                            print(response[print_idx:], end='', flush=True)
                            print_idx = len(response)
                    print()
                else:
                    response, _ = inference(
                        model, template, stream=args.stream and args.verbose, verbose=args.verbose, **kwargs)
                label = data.pop('response', None)
                obj = {
                    'system': kwargs['system'],
                    'query': kwargs['query'],
                    'response': response,
                    'label': label,
                    'history': kwargs['history'],
                }
                for media_key in MediaTag.media_keys.values():
                    media_files = kwargs.get(media_key)
                    if media_files is not None:
                        obj[media_key] = media_files
                if jsonl_path is not None:
                    append_to_jsonl(jsonl_path, obj)
                result.append(obj)
                if args.verbose:
                    print()
                    print(f'[LABELS]{label}')
                    for media_key in MediaTag.media_keys.values():
                        media_files = kwargs.get(media_key)
                        if media_files is not None:
                            print(f'[{media_key.upper()}]{media_files}')
                    print('-' * 50, flush=True)

    if jsonl_path is not None:
        logger.info(f'save_result_path: {jsonl_path}')
    if not args.eval_human and args.show_dataset_sample == 10:  # is default
        logger.info('You can set `--show_dataset_sample -1` to perform inference on the entire dataset.')
    return {'result': result}


infer_main = get_main(InferArguments, llm_infer)
merge_lora_main = get_main(InferArguments, merge_lora)
