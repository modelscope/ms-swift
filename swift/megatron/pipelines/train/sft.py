# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import asdict
from typing import List, Optional, Union

import torch
from transformers.utils import is_torch_npu_available

from swift.megatron.arguments import MegatronSftArguments
from swift.megatron.trainers import MegatronEmbeddingTrainer, MegatronRerankerTrainer, MegatronTrainer
from swift.pipelines import SwiftSft
from swift.utils import get_logger, is_last_rank, plot_images

if is_torch_npu_available():
    # Enable Megatron on Ascend NPU
    from mindspeed.megatron_adaptor import repatch
else:
    repatch = None

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronSftArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        if args.task_type == 'embedding':
            return MegatronEmbeddingTrainer(self.args, self.template)
        elif args.task_type in {'reranker', 'generative_reranker'}:
            return MegatronRerankerTrainer(self.args, self.template)
        else:
            return MegatronTrainer(self.args, self.template)

    def _set_seed(self):
        pass

    def __init__(self, args: Optional[Union[List[str], MegatronSftArguments]] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        if repatch is not None:
            if args.attention_backend != 'local':
                # MindSpeed requires passing `use_flash_attn` to Megatron
                # to enable flash attention on Ascend NPU.
                args.use_flash_attn = True
            megatron_args = asdict(self.args)
            repatch(megatron_args)
        template_cls = args.template_meta.template_cls
        if args.model_meta.is_multimodal and template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self.model, self.processor = args.get_model_processor(**kwargs, download_model=args.mcore_model is None)
        self._prepare_template()
        args.save_args(args.output_dir)
        self.template.use_megatron = True

    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        args.init_iters(train_dataset, val_dataset)
        trainer = self.prepare_trainer()
        try:
            trainer.train(train_dataset, val_dataset)
        finally:
            # Visualization
            if is_last_rank():
                images_dir = os.path.join(args.output_dir, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Optional[Union[List[str], MegatronSftArguments]] = None):
    return MegatronSft(args).main()
