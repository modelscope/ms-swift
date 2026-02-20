# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from typing import List, Optional, Union

import torch.distributed as dist
from transformers.utils import strtobool

from swift.megatron.arguments import MegatronExportArguments
from swift.megatron.convert import test_convert_precision
from swift.megatron.model import get_mcore_model
from swift.megatron.utils import load_mcore_checkpoint, prepare_mcore_model, save_mcore_checkpoint
from swift.pipelines import SwiftPipeline, prepare_model_template
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger, is_last_rank

logger = get_logger()


class MegatronExport(SwiftPipeline):
    args_class = MegatronExportArguments
    args: args_class

    def run(self):
        os.environ['DISABLE_MP_DDP'] = 'true'
        args = self.args
        if args.to_hf:
            self.convert_mcore2hf()
        elif args.to_mcore:
            self.convert_hf2mcore()

    def convert_mcore2hf(self) -> None:
        args = self.args
        download_model = args.model is not None
        _, template = prepare_model_template(args, load_model=False, download_model=download_model)
        self.processor = template.processor
        hf_config = self.processor.model_info.config
        mg_model = get_mcore_model(args, hf_config)[0]
        logger.info('Megatron model created successfully.')
        bridge = args.megatron_model_meta.bridge_cls(args)
        if args.mcore_model is not None:
            load_mcore_checkpoint(args, [mg_model], load_arg='mcore_model')
        elif args.model is not None:
            bridge.load_weights([mg_model], args.model_info.model_dir)
        else:
            raise ValueError('Please specify `--mcore_model` or `--model`.')
        if args.adapters or args.mcore_adapter is not None:
            peft_model = prepare_mcore_model(args, mg_model)
            if args.mcore_adapter is not None:
                load_mcore_checkpoint(args, [mg_model], load_arg='mcore_adapter')
            elif args.adapters:
                assert len(args.adapters) == 1, 'Currently only support one adapter'
                bridge.load_weights([mg_model], args.adapters[0], is_peft_format=True)
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Converting weights and saving the model...')
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        bridge.save_weights([mg_model],
                            args.output_dir,
                            is_peft_format=save_peft_format,
                            processor=self.processor,
                            hf_config=hf_config)
        args_path = os.path.join(args.mcore_adapter or args.mcore_model or args.model, 'args.json')
        if os.path.exists(args_path):
            if is_last_rank():
                shutil.copy(args_path, os.path.join(args.output_dir, 'args.json'))
        else:
            args.save_args(args.output_dir)
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                if save_peft_format:
                    kwargs = {'adapters': [args.output_dir]}
                else:
                    kwargs = {'model': args.output_dir, 'torch_dtype': None}
                device_map = args.device_map or 'auto'
                hf_model, template = prepare_model_template(
                    args, device_map=device_map, **kwargs) if is_last_rank() else (None, template)
            test_convert_precision(args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)
            dist.barrier()

    def convert_hf2mcore(self) -> None:
        args = self.args
        download_model = args.model is not None
        _, template = prepare_model_template(args, load_model=False, download_model=download_model)
        self.processor = template.processor
        hf_config = self.processor.model_info.config
        mg_model = get_mcore_model(args, hf_config)[0]
        logger.info('Megatron model created successfully.')
        bridge = args.megatron_model_meta.bridge_cls(args)
        if args.model is not None:
            bridge.load_weights([mg_model], args.model_info.model_dir)
        elif args.mcore_model is not None:
            load_mcore_checkpoint(args, [mg_model], load_arg='mcore_model')
        else:
            raise ValueError('Please specify `--mcore_model` or `--model`.')
        dist.barrier()
        if args.adapters or args.mcore_adapter is not None:
            peft_model = prepare_mcore_model(args, mg_model)
            if args.adapters:
                assert len(args.adapters) == 1, 'Currently only support one adapter'
                bridge.load_weights([mg_model], args.adapters[0], is_peft_format=True)
            elif args.mcore_adapter is not None:
                load_mcore_checkpoint(args, [mg_model], load_arg='mcore_adapter')
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Successfully transferred HF model weights to MG model.')
        _test_convert_precision = strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0'))
        if not _test_convert_precision:
            args.save_args(args.output_dir)
            logger.info('Saving the model...')
            save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
            save_mcore_checkpoint(args, [mg_model], is_peft_format=save_peft_format)
        # hf_model does not support loading args.mcore_adapter, so test_convert_precision cannot be performed
        support_convert_precision = args.mcore_adapter is None
        if args.test_convert_precision:
            if support_convert_precision:
                with disable_safe_ddp_context_use_barrier():
                    device_map = args.device_map or 'auto'
                    hf_model, template = prepare_model_template(
                        args, device_map=device_map) if is_last_rank() else (None, template)
                test_convert_precision(args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)
                dist.barrier()
            else:
                logger.warning('Skip test_convert_precision because `--mcore_adapter` is specified.')


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
