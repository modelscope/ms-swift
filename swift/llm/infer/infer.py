# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Literal, Optional

import json
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from swift.hub import default_hub, hub
from swift.llm import DatasetLoader, InferArguments, standard_keys, standard_tags
from swift.llm.infer.base import InferFramework
from swift.llm.infer.transformers import TransformersFramework
from swift.tuners import Swift
from swift.utils import append_to_jsonl, get_logger, get_main, read_multi_line, seed_everything

logger = get_logger()


def read_media_file(infer_kwargs: Dict[str, Any], infer_media_type: Literal['none', 'round', 'dialogue', 'interleave'],
                    media_type: Literal['image', 'video', 'audio'], query: str) -> None:
    if infer_media_type == 'none':
        return

    def _input_media(media_type: Literal['image', 'video', 'audio']) -> None:
        media_key = standard_keys[media_type]
        media_files = infer_kwargs.get(media_key) or []
        a_an = 'an' if media_type[0] in {'i', 'a'} else 'a'
        text = f'Input {a_an} {media_type} path or URL <<< '
        media_files += [input(text) or None]
        infer_kwargs[media_key] = media_files

    if infer_media_type == 'interleave':
        media_tags = re.findall('|'.join(list(standard_tags.values())), query)
        standard_tags_r = {v: k for k, v in standard_tags.items()}
        for tag in media_tags:
            media_type = standard_tags_r[tag]
            _input_media(media_type)
        return

    media_key = standard_keys[media_type]
    media_files = infer_kwargs.get(media_key) or []
    if infer_media_type == 'round' or len(media_files) == 0:
        _input_media(media_type)


def eval_human(framework: InferFramework, args: InferArguments, jsonl_path: str, result: List[Dict[str, Any]]):
    input_mode: Literal['S', 'M'] = 'S'
    logger.info('Input `exit` or `quit` to exit the conversation.')
    logger.info('Input `multi-line` to switch to multi-line input mode.')
    logger.info('Input `reset-system` to reset the system and clear the history.')
    if framework.template.support_multi_round:
        logger.info('Input `clear` to clear the history.')
    else:
        logger.info('The current template only supports single-round dialogues.')

    lora_request = None
    if args.vllm_enable_lora:
        assert len(args.lora_request_list) == 1
        lora_request = args.lora_request_list[0]

    messages = []
    infer_kwargs = {}
    if args.infer_media_type != 'none':
        logger.info('Please enter the conversation content first, followed by the path to the multimedia file.')
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
            messages = []
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
            messages = []
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
        if not framework.template.support_multi_round:
            messages = []
            infer_kwargs = {}

        read_media_file(infer_kwargs, args.infer_media_type, args.media_type, query)
        if system is None and framework.template.use_default_system:
            system = framework.template.default_system

        messages.append({'role': 'user', 'content': query})
        request_list = [{'messages': messages, 'system': system, **infer_kwargs}]
        if args.stream:
            gen = framework.inference_stream(request_list, lora_request=lora_request)
            print_idx = 0
            response = None
            for resp_list in gen:
                response = resp_list[0]['response']
                if len(response) > print_idx:
                    print(response[print_idx:], end='', flush=True)
                    print_idx = len(response)
            assert response is not None
            messages.append({'role': 'assistant', 'content': response})
            print()
        else:
            resp_list = framework.inference(request_list, lora_request=lora_request)
            response = resp_list[0]['response']
            messages.append({'role': 'assistant', 'content': response})
            print(response)

        print('-' * 50)
        obj = {
            'system': system,
            'messages': messages,
        }
        for media_key in standard_keys.values():
            media_files = infer_kwargs.get(media_key)
            if media_files is not None:
                obj[media_key] = media_files
        if jsonl_path is not None:
            append_to_jsonl(jsonl_path, obj)
        result.append(obj)


def eval_dataset(framework: InferFramework, args: InferArguments, jsonl_path: str, result: List[Dict[str, Any]]):
    dataset_kwargs = {
        'dataset_seed': args.dataset_seed,
        'check_dataset_strategy': args.check_dataset_strategy,
        'model_name': args.model_name,
        'model_author': args.model_author
    }
    if len(args.val_dataset) > 0:
        _, val_dataset = DatasetLoader.load_dataset(args.val_dataset, 1.0, **dataset_kwargs)
    else:
        _, val_dataset = DatasetLoader.load_dataset(args.dataset, args.dataset_test_ratio, **dataset_kwargs)
    assert val_dataset is not None
    if 0 <= args.show_dataset_sample < val_dataset.shape[0]:
        random_state = np.random.RandomState(args.dataset_seed)
        logger.info(f'show_dataset_sample: {args.show_dataset_sample}')
        val_dataset = DatasetLoader.sample_dataset(val_dataset, args.show_dataset_sample, random_state)
    logger.info(f'val_dataset: {val_dataset}')
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
        if system is None and framework.template.use_default_system:
            system = framework.template.default_system
        request['system'] = system
        for media_key in standard_keys.values():
            media_files = data.get(media_key)
            if media_files is not None:
                request[media_key] = media_files
        request['truncation_strategy'] = args.truncation_strategy
        request_list.append(request)
    resp_list = framework.inference(request_list, use_tqdm=True)
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
        for media_key in standard_keys.values():
            media_files = request.get(media_key)
            if media_files is not None:
                obj[media_key] = media_files
        if jsonl_path is not None:
            append_to_jsonl(jsonl_path, obj)
        result.append(obj)


def llm_infer(args: InferArguments) -> Dict[str, List[Dict[str, Any]]]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    if args.merge_lora:
        merge_lora(args, device_map=args.merge_device_map)

    if args.infer_backend == 'vllm':
        from .vllm import VllmEngine
        framework = VllmEngine(args)
    elif args.infer_backend == 'lmdeploy':
        from .lmdeploy import LMDeployFramework
        framework = LMDeployFramework(args)
    elif args.infer_backend == 'pt':
        framework = TransformersFramework(args)
    else:
        raise ValueError(f'Unsupported backend: {args.infer_backend}')

    # Inference
    result: List[Dict[str, Any]] = []
    jsonl_path = None
    if args.save_result:
        if args.result_dir:
            result_dir = args.result_dir
        else:
            result_dir = args.ckpt_dir
            if result_dir is None:
                result_dir = framework.llm_engine.model_dir
            if result_dir is not None:
                result_dir = os.path.join(result_dir, 'infer_result')
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)
            time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
            jsonl_path = os.path.join(result_dir, f'{time}.jsonl')

    if args.eval_human:
        eval_human(framework, args, jsonl_path, result)
    else:
        eval_dataset(framework, args, jsonl_path, result)
    if jsonl_path is not None:
        logger.info(f'save_result_path: {jsonl_path}')
    return {'result': result}


infer_main = get_main(InferArguments, llm_infer)
