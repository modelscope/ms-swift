# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union

from megatron.core.enums import ModelType
from megatron.training import pretrain

from swift.llm.train import SwiftSft
from swift.utils import get_logger, is_master, plot_images
from ..argument import MegatronTrainArguments
from ..utils import patch_megatron_tokenizer
from .patcher import patch_megatron_data_collator, patch_training_log
from .utils import build_streaming_dataloader, forward_step, get_swift_datasets_provider

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronTrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], MegatronTrainArguments, None] = None) -> None:
        self.train_msg = {}
        super(SwiftSft, self).__init__(args)
        args = self.args
        _, self.processor = args.get_model_processor(load_model=False)
        patch_megatron_tokenizer(self.processor)
        args.init_model_args(self.processor.model_info.config)
        self._prepare_template()
        self.template.use_megatron = True
        args.save_args(args.save)

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        data_collator = self.template.data_collator
        if args.streaming:
            train_dataset = build_streaming_dataloader(args, train_dataset, data_collator)
            if val_dataset is not None:
                val_dataset = build_streaming_dataloader(args, val_dataset, data_collator)
        datasets_provider = get_swift_datasets_provider(train_dataset, val_dataset)
        datasets_provider.is_distributed = True

        logging_path = os.path.join(args.save, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        with patch_training_log(), patch_megatron_data_collator(data_collator):
            pretrain(
                datasets_provider,
                args.megatron_model_meta.model_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                args_defaults=args.extra_args)

        # Visualization
        if is_master():
            images_dir = os.path.join(args.save, 'images')
            logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronSft(args).main()
