# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.llm import ExportArguments, SwiftPipeline
from swift.tuners import swift_to_peft_format
from swift.utils import get_logger
from .merge_lora import merge_lora
from .ollama import export_to_ollama
from .quant import quantize_model

logger = get_logger()


class SwiftExport(SwiftPipeline):
    args_class = ExportArguments
    args: args_class

    def run(self):
        args = self.args
        if args.to_peft_format:
            args.adapters[0] = swift_to_peft_format(args.adapters[0], args.output_dir)
        if args.merge_lora:
            output_dir = args.output_dir
            if args.to_peft_format or args.quant_method or args.to_ollama or args.push_to_hub:
                args.output_dir = None
            merge_lora(args)
            args.output_dir = output_dir  # recover
        if args.quant_method:
            quantize_model(args)
        elif args.to_ollama:
            export_to_ollama(args)
        elif args.to_mcore:
            from swift.megatron import convert_hf2mcore
            convert_hf2mcore(args)
        elif args.to_hf:
            from swift.megatron import convert_mcore2hf
            convert_mcore2hf(args)
        elif args.push_to_hub:
            model_dir = args.adapters and args.adapters[0] or args.model_dir
            assert model_dir, f'model_dir: {model_dir}'
            args.hub.push_to_hub(
                args.hub_model_id,
                model_dir,
                token=args.hub_token,
                private=args.hub_private_repo,
                commit_message=args.commit_message)


def export_main(args: Union[List[str], ExportArguments, None] = None):
    return SwiftExport(args).main()
