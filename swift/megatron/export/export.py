# Copyright (c) Alibaba, Inc. and its affiliates.
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
        _, template = prepare_model_template(args, load_model=False)
        self.processor = template.processor
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
        megatron_model_meta = args.megatron_model_meta
        extra_args_provider = megatron_model_meta.extra_args_provider
        initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args.extra_args)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        mg_model = megatron_model_meta.model_provider(pre_process=pre_process, post_process=post_process)
        with patch_load_base_checkpoint():
            load_checkpoint([mg_model], None, None, strict=True)
        if args.adapter_load is not None:
            prepare_mcore_model(mg_model)
            with adapter_state_dict_context():
                load_checkpoint([mg_model], None, None, load_arg='adapter_load', strict=False)
        logger.info('Converting weights and saving the model...')
        bridge = megatron_model_meta.bridge_cls()
        save_peft_format = args.train_type == 'lora' and not args.merge_lora
        bridge.save_weights([mg_model], args.save, is_peft_format=save_peft_format)
        if is_last_rank():
            args_path = os.path.join(os.path.dirname(args.save), 'args.json')
            if os.path.exists(args_path):
                shutil.copy(args_path, os.path.join(args.save, 'args.json'))
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                if save_peft_format:
                    kwargs = {'adapters': [args.save]}
                else:
                    kwargs - {'model': args.save}
                hf_model = prepare_model_template(args, device_map='cpu', **kwargs)[0] if is_last_rank() else None
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
            dist.barrier()

    def convert_hf2mcore(self) -> None:
        args = self.args
        _, template = prepare_model_template(args, load_model=False)
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
        bridge.load_weights(mg_model, args.model_info.model_dir)
        dist.barrier()
        if args.adapters:
            prepare_mcore_model(mg_model)
            assert len(args.adapters) == 1, 'Currently only support one adapter'
            bridge.load_weights(mg_model, args.adapters[0], is_peft_format=True)
        logger.info('Successfully transferred HF model weights to MG model.')
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                hf_model = prepare_model_template(args, device_map='cpu')[0] if is_last_rank() else None
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
            dist.barrier()
        if is_last_rank():
            args.save_args(args.save)
        logger.info('Saving the model...')
        with adapter_state_dict_context():
            mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info_if(f'Successfully saved Megatron model weights in `{args.save}`.', cond=is_last_rank())


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
