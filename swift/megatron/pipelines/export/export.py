# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from typing import List, Optional, Union

import torch.distributed as dist
from megatron.core import mpu
from megatron.training import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint
from transformers.utils import strtobool

from swift.megatron.arguments import MegatronExportArguments
from swift.megatron.convert import test_convert_precision
from swift.megatron.utils import adapter_state_dict_context, patch_load_base_checkpoint, prepare_mcore_model
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
        args.init_model_args(self.tokenizer, hf_config)
        megatron_model_meta = args.megatron_model_meta
        extra_args_provider = megatron_model_meta.extra_args_provider
        initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args.extra_args)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        mg_model = megatron_model_meta.model_provider(pre_process=pre_process, post_process=post_process)
        bridge = megatron_model_meta.bridge_cls()
        if args.load is not None:
            with patch_load_base_checkpoint():
                load_checkpoint([mg_model], None, None, strict=True)
        elif args.model is not None:
            bridge.load_weights(mg_model, args.model_info.model_dir)
        else:
            raise ValueError('Please specify `--load` or `--model`.')
        if args.adapter_load is not None:
            peft_model = prepare_mcore_model(mg_model)
            with adapter_state_dict_context():
                load_checkpoint([mg_model], None, None, load_arg='adapter_load', strict=False)
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Converting weights and saving the model...')
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        bridge.save_weights([mg_model],
                            args.save,
                            is_peft_format=save_peft_format,
                            processor=self.processor,
                            config=hf_config)
        args_path = os.path.join(args.adapter_load or args.load or args.model, 'args.json')
        if os.path.exists(args_path):
            if is_last_rank():
                shutil.copy(args_path, os.path.join(args.save, 'args.json'))
        else:
            args.save_args(args.save)
        logger.info(f'Successfully saved HF model weights in `{args.save}`.')
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                if save_peft_format:
                    kwargs = {'adapters': [args.save]}
                else:
                    kwargs = {'model': args.save, 'torch_dtype': None}
                device_map = args.device_map or 'auto'
                hf_model, template = prepare_model_template(
                    args, device_map=device_map, **kwargs) if is_last_rank() else (None, template)
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
            dist.barrier()

    def convert_hf2mcore(self) -> None:
        args = self.args
        download_model = args.model is not None
        _, template = prepare_model_template(args, load_model=False, download_model=download_model)
        self.processor = template.processor
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
        megatron_model_meta = args.megatron_model_meta
        extra_args_provider = megatron_model_meta.extra_args_provider
        initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args.extra_args)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        mg_model = megatron_model_meta.model_provider(pre_process=pre_process, post_process=post_process)
        logger.info('Megatron model created successfully.')
        bridge = megatron_model_meta.bridge_cls()
        if args.model is not None:
            bridge.load_weights(mg_model, args.model_info.model_dir)
        elif args.load is not None:
            with patch_load_base_checkpoint():
                load_checkpoint([mg_model], None, None, strict=True)
        else:
            raise ValueError('Please specify `--load` or `--model`.')
        dist.barrier()
        if args.adapters or args.adapter_load is not None:
            peft_model = prepare_mcore_model(mg_model)
            if args.adapters:
                assert len(args.adapters) == 1, 'Currently only support one adapter'
                bridge.load_weights(mg_model, args.adapters[0], is_peft_format=True)
            elif args.adapter_load is not None:
                with adapter_state_dict_context():
                    load_checkpoint([mg_model], None, None, load_arg='adapter_load', strict=False)
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Successfully transferred HF model weights to MG model.')
        _test_convert_precision = strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0'))
        if not _test_convert_precision:
            args.save_args(args.save)
            logger.info('Saving the model...')
            save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
            with adapter_state_dict_context(is_peft_format=save_peft_format):
                mg_save_checkpoint(1, [mg_model], None, None, 0)
            logger.info_if(f'Successfully saved Megatron model weights in `{args.save}`.', cond=is_last_rank())
        # hf_model does not support loading args.adapter_load, so test_convert_precision cannot be performed
        support_convert_precision = args.adapter_load is None
        if args.test_convert_precision:
            if support_convert_precision:
                with disable_safe_ddp_context_use_barrier():
                    device_map = args.device_map or 'auto'
                    hf_model, template = prepare_model_template(
                        args, device_map=device_map) if is_last_rank() else (None, template)
                test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
                dist.barrier()
            else:
                logger.warning('Skip test_convert_precision because `--adapter_load` is specified.')


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
