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
            args.ckpt_dir = swift_to_peft_format(args.ckpt_dir, args.output_dir)
        elif args.merge_lora:
            merge_lora(args)
        elif args.quant_method is not None:
            quantize_model(args)
        elif args.to_ollama:
            export_to_ollama(args)
        elif args.push_to_hub:
            ckpt_dir = args.ckpt_dir or args.model
            assert ckpt_dir is not None, 'You need to specify `ckpt_dir`.'
            args.hub.push_to_hub(
                args.hub_model_id,
                ckpt_dir,
                token=args.hub_token,
                private=args.hub_private_repo,
                commit_message=args.commit_message)


def export_main(args: Union[List[str], ExportArguments, None] = None):
    return SwiftExport(args).main()
