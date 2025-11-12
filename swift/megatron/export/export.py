# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import List, Optional, Union

import torch.distributed as dist
from megatron.core import mpu
from megatron.training import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint

from swift.llm import SwiftPipeline, prepare_model_template
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger, is_last_rank
from ..argument import MegatronExportArguments
from ..convert import test_convert_precision
from ..utils import adapter_state_dict_context, patch_load_base_checkpoint, prepare_mcore_model

logger = get_logger()


class MegatronExport(SwiftPipeline):
    args_class = MegatronExportArguments
    args: args_class

    def run(self):
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
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
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
        save_peft_format = args.train_type == 'lora' and not args.merge_lora
        bridge.save_weights([mg_model], args.save, is_peft_format=save_peft_format)
        if is_last_rank():
            args_path = os.path.join(args.adapter_load or args.load or args.model, 'args.json')
            if os.path.exists(args_path):
                shutil.copy(args_path, os.path.join(args.save, 'args.json'))
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                if save_peft_format:
                    kwargs = {'adapters': [args.save]}
                else:
                    kwargs = {'model': args.save}
                hf_model, template = prepare_model_template(
                    args, device_map='cpu', **kwargs) if is_last_rank() else (None, template)
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
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                hf_model, template = prepare_model_template(
                    args, device_map='cpu') if is_last_rank() else (None, template)
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
            dist.barrier()
        args.save_args(args.save)
        logger.info('Saving the model...')
        save_peft_format = args.train_type == 'lora' and not args.merge_lora
        with adapter_state_dict_context(is_peft_format=save_peft_format):
            mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info_if(f'Successfully saved Megatron model weights in `{args.save}`.', cond=is_last_rank())


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
