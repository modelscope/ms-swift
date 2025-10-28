# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from megatron.training import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.training.checkpointing import save_checkpoint as mg_save_checkpoint

from swift.llm import SwiftPipeline, prepare_model_template
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger, is_master
from ..argument import MegatronExportArguments
from ..convert import test_convert_precision
from ..utils import prepare_mcore_model

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

        mg_model = megatron_model_meta.model_provider()
        load_checkpoint([mg_model], None, None, strict=True)
        logger.info('Converting weights and saving the model...')
        bridge = megatron_model_meta.bridge_cls()
        bridge.save_weights([mg_model], args.save)
        logger.info(f'Successfully saved HF model weights in `{args.save}`.')
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                hf_model = prepare_model_template(args, model=args.save, device_map='cpu')[0] if is_master() else None
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)

    def convert_hf2mcore(self) -> None:
        args = self.args
        _, template = prepare_model_template(args, load_model=False)
        self.processor = template.processor
        args.init_model_args(self.tokenizer, self.processor.model_info.config)
        megatron_model_meta = args.megatron_model_meta
        extra_args_provider = megatron_model_meta.extra_args_provider
        initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=args.extra_args)

        mg_model = megatron_model_meta.model_provider()
        logger.info('Megatron model created successfully.')
        bridge = megatron_model_meta.bridge_cls()
        bridge.load_weights(mg_model, args.model_info.model_dir)
        logger.info('Successfully transferred HF model weights to MG model.')
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                hf_model = prepare_model_template(args, device_map='cpu')[0] if is_master() else None
            test_convert_precision(hf_model, mg_model, template, args.test_convert_dtype)
        args.save_args(args.save)
        logger.info('Saving the model...')
        mg_save_checkpoint(1, [mg_model], None, None, 0)
        logger.info(f'Successfully saved Megatron model weights in `{args.save}`.')


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
