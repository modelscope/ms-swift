# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile

from swift.llm import ExportArguments, prepare_model_template, save_checkpoint
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()


def merge_lora(args: ExportArguments, device_map=None, replace_if_exists=False) -> None:
    if replace_if_exists:
        logger.info(f'replace_if_exists: {replace_if_exists}')
    assert args.quant_method is None, (f'args.quant_method: {args.quant_method}, '
                                       'quantized model and does not support merge-lora.')

    output_dir = getattr(args, 'output_dir', None) or f'{args.adapters[0]}-merged'
    if os.path.exists(output_dir) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {output_dir}, '
                    'skipping the saving process. '
                    'you can pass `replace_if_exists=True` to overwrite it.')
    else:
        origin_device_map = args.device_map
        args.device_map = device_map or args.device_map
        logger.info(f'merge_device_map: {device_map}')
        if args.use_merge_kit:
            base_model = args.model
            if not os.path.exists(base_model):
                base_model = args.hub.download_model(base_model, revision=args.model_revision)
            if not os.path.exists(args.instruct_model):
                args.instruct_model = args.hub.download_model(
                    args.instruct_model, revision=args.instruct_model_revision)
            args.model = args.instruct_model
        model, template = prepare_model_template(args)
        logger.info('Merge LoRA...')
        Swift.merge_and_unload(model)
        model = model.model
        logger.info('Saving merged weights...')

        save_checkpoint(
            model,
            template.processor,
            output_dir,
            safe_serialization=args.safe_serialization,
            model_dirs=args.adapters,
            max_shard_size=args.max_shard_size,
            additional_saved_files=model.model_meta.additional_saved_files)
        logger.info(f'Successfully merged LoRA and saved in {output_dir}.')
        args.device_map = origin_device_map

    args.model = output_dir
    args.adapters = []
    if args.use_merge_kit:
        tempdir = tempfile.gettempdir()
        mergekit_path = os.path.join(output_dir, 'mergekit')
        merge_yaml = args.merge_yaml.replace('{merged_model}', output_dir).replace('{instruct_model}',
                                                                                   args.instruct_model).replace(
                                                                                       '{base_model}', base_model)
        try:
            yamlfile = os.path.join(tempdir, 'mergekit.yaml')
            with open(yamlfile, 'w') as f:
                f.write(merge_yaml)
            logger.info(f'Merging with config: {merge_yaml}')
            os.system(f'mergekit-yaml {yamlfile} {mergekit_path}')
            logger.info(f'Merge complete with path: {mergekit_path}')
        finally:
            if tempdir:
                shutil.rmtree(tempdir, ignore_errors=True)
