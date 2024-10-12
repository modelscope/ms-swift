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


def save_checkpoint(model: Optional[PreTrainedModel],
                    tokenizer: PreTrainedTokenizerBase,
                    model_cache_dir: str,
                    ckpt_dir: Optional[str],
                    target_dir: str,
                    *,
                    save_safetensors: bool = True,
                    sft_args_kwargs: Optional[Dict[str, Any]] = None,
                    **kwargs) -> None:
    if sft_args_kwargs is None:
        sft_args_kwargs = {}
    if model is not None:
        model.save_pretrained(target_dir, safe_serialization=save_safetensors)
    if hasattr(tokenizer, 'processor'):
        tokenizer.processor.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    model_type = getattr(tokenizer, 'model_type')
    fname_list = ['generation_config.json', 'preprocessor_config.json']
    if model_type is not None:
        fname_list += kwargs.get('additional_saved_files', [])

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
    assert args.train_type in args.adapters_can_be_merged, 'Only supports lora & llamapro series models'
    assert not args.is_quant_model(), f'{args.model_type} is a quantized model and does not support merge-lora.'
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
        if args.use_merge_kit:
            base_model_id_or_path = args.model_id_or_path
            if not os.path.exists(args.instruct_model_id_or_path):
                args.instruct_model_id_or_path = default_hub.download_model(
                    args.instruct_model_id_or_path, revision=args.instruct_model_revision)
            args.model_id_or_path = args.instruct_model_id_or_path
        model, template = TransformersFramework.prepare_model_template_hf(args)
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
            sft_args_kwargs={'dtype': args.dtype},
            additional_saved_files=args.get_additional_saved_files())
        logger.info(f'Successfully merged LoRA and saved in {merged_lora_path}.')
        if args.use_merge_kit:
            tempdir = tempfile.gettempdir()
            mergekit_path = merged_lora_path + '-mergekit'
            merge_yaml = args.merge_yaml.replace('{merged_model}', merged_lora_path).replace(
                '{instruct_model}', args.instruct_model_id_or_path).replace('{base_model}', base_model_id_or_path)
            try:
                yamlfile = os.path.join(tempdir, 'mergekit.yaml')
                with open(yamlfile, 'w') as f:
                    f.write(merge_yaml)
                os.system(f'mergekit-yaml {yamlfile} {mergekit_path}')
            finally:
                if tempdir:
                    shutil.rmtree(tempdir, ignore_errors=True)

    logger.info("Setting args.train_type: 'full'")
    logger.info(f'Setting args.ckpt_dir: {merged_lora_path}')
    args.train_type = 'full'
    args.ckpt_dir = merged_lora_path
    return merged_lora_path


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
merge_lora_main = get_main(InferArguments, merge_lora)
