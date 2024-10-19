import os
import shutil
from typing import Any, Dict, Optional

import json
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from swift.hub import default_hub
from swift.llm import InferArguments
from swift.tuners import Swift
from swift.utils import get_logger

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
    assert args.quant_method is None, f'{args.model_type} is a quantized model and does not support merge-lora.'
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


merge_lora_main = get_main(InferArguments, merge_lora)
