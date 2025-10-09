# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from typing import List, Optional, Union

import torch

from swift.llm import TEMPLATE_MAPPING
from swift.llm.train import SwiftSft
from swift.utils import get_logger, is_master, plot_images
from ..argument import MegatronTrainArguments
from ..trainers import MegatronTrainer
from ..utils import patch_megatron_tokenizer
from .utils import build_streaming_dataloader

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronTrainArguments
    args: args_class

    def prepare_trainer(self):
        return MegatronTrainer(self.args, self.template)

    def __init__(self, args: Optional[Union[List[str], MegatronTrainArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        template_cls = TEMPLATE_MAPPING[args.template].template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs)
        self._prepare_template()
        patch_megatron_tokenizer(self.processor)
        args.init_model_args(self.processor, self.processor.model_info.config)
        args.save_args(args.save)
        self.template.use_megatron = True
        self.trainer = self.prepare_trainer()

    def _get_data_collator(self):
        args = self.args
        data_collator = self.template.data_collator
        padding_to = None
        if args.tensor_model_parallel_size > 1 and args.sequence_parallel:
            padding_to = args.tensor_model_parallel_size
        if args.context_parallel_size > 1:
            padding_to = (padding_to or 1) * args.context_parallel_size
        if args.fp8_format:
            padding_to = max((padding_to or 1) * 8, 16)
        logger.info(f'padding_to: {padding_to}')
        data_collator = partial(data_collator, padding_to=padding_to)
        return data_collator

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        data_collator = self._get_data_collator()

        if args.streaming:
            train_dataset = build_streaming_dataloader(args, train_dataset, data_collator)
            if val_dataset is not None:
                val_dataset = build_streaming_dataloader(args, val_dataset, data_collator)

        try:
            self.trainer.train(train_dataset, val_dataset, data_collator)
        finally:
            # Visualization
            if is_master():
                images_dir = os.path.join(args.save, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Optional[Union[List[str], MegatronTrainArguments]] = None):
    return MegatronSft(args).main()
