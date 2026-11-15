# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from typing import List, Optional, Union

from swift.pipelines import SwiftPipeline
from swift.pipelines.train.sft import SwiftSft
from swift.utils import append_to_jsonl, get_logger, is_master
from .arguments import FSDPTurboSftArguments
from .trainer import FSDPTurboTrainer

logger = get_logger()


class FSDPTurboSft(SwiftSft):
    args_class = FSDPTurboSftArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], FSDPTurboSftArguments]] = None) -> None:
        SwiftPipeline.__init__(self, args)
        self.train_msg = {}
        template_cls = self.args.template_meta.template_cls
        if self.args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = self.args.get_model_processor(**kwargs)
        self._prepare_template()

    def run(self):
        train_dataset, val_dataset = self._prepare_dataset()
        self.args.save_args()
        self.model = None
        self.template.model = None
        trainer = FSDPTurboTrainer(self.args, self.template, self.processor, train_dataset, val_dataset)
        try:
            trainer.setup()
            trainer.train()
            self.train_msg.update(trainer.result)
            if is_master():
                append_to_jsonl(f'{self.args.output_dir}/logging.jsonl', self.train_msg, strict=False)
            return self.train_msg
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def fsdpturbo_sft_main(args: Optional[Union[List[str], FSDPTurboSftArguments]] = None):
    return FSDPTurboSft(args).main()
