# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from functools import partial
from typing import List, Union

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, pretrain, training

from swift.llm.train import SwiftSft
from swift.utils import get_logger, is_master, plot_images
from ..argument import MegatronTrainArguments
from ..utils import patch_megatron_tokenizer
from .patcher import patch_megatron_data_collator
from .utils import build_streaming_dataloader, get_batch, get_swift_datasets_provider

logger = get_logger()

stimer = StragglerDetector()


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

    @contextmanager
    def _get_train_iters(self, train_dataset):
        from megatron.training import training
        origin_initialize_megatron = training.initialize_megatron

        def initialize_megatron(*_args, **kwargs):
            res = origin_initialize_megatron(*_args, **kwargs)
            args = get_args()
            if args.train_iters is None and hasattr(train_dataset, '__len__'):
                data_parallel_size = mpu.get_data_parallel_world_size()
                step_batch_size = \
                    args.micro_batch_size * data_parallel_size
                dataset_sample = len(train_dataset) // step_batch_size * step_batch_size
                args.train_iters = (dataset_sample * args.max_epochs // args.global_batch_size) + 1
            return res

        training.initialize_megatron = initialize_megatron
        try:
            yield
        finally:
            training.initialize_megatron = origin_initialize_megatron

    @staticmethod
    def new_cyclic_iter(iter):
        args = get_args()
        max_epochs = args.max_epochs
        i = 0
        while True:
            if getattr(args, 'is_training', False):
                if max_epochs and i >= max_epochs:
                    logger.info(f'Training of {i} epochs has been completed, the training has finished.')
                    break
                logger.info(f'The training of Epoch {i} starts...')
            for x in iter:
                yield x
            i += 1

    @staticmethod
    @contextmanager
    def _training_context():
        args = get_args()
        args.is_training = True
        try:
            yield
        finally:
            args.is_training = False

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        return self._train_step_origin(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config)

    def _patch_train_step(self):
        # support max_epochs
        def train_step(*args, **kwargs):
            with self._training_context():
                try:
                    return self.train_step(*args, **kwargs)
                except StopIteration:
                    return {}, True, True, True, 0, None, None

        self._train_step_origin = training.train_step
        training.train_step = train_step
        training.cyclic_iter = MegatronSft.new_cyclic_iter

    def forward_step(self, data_iterator, model):
        from pretrain_gpt import loss_func

        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        global stimer
        with stimer(bdata=True):
            data = get_batch(data_iterator)
        if not data:
            raise StopIteration
        timers('batch-generator').stop()

        with stimer:
            output_tensor = model(**data)
        labels = data.get('labels')
        loss_mask = None if labels is None else (labels != -100).float()
        return output_tensor, partial(loss_func, loss_mask)

    def run(self):
        args = self.args
        self._patch_train_step()

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
        try:
            with patch_megatron_data_collator(data_collator), self._get_train_iters(train_dataset):
                extra_args_provider = args.megatron_model_meta.extra_args_provider
                pretrain(
                    datasets_provider,
                    args.megatron_model_meta.model_provider,
                    ModelType.encoder_or_decoder,
                    self.forward_step,
                    extra_args_provider=extra_args_provider,
                    args_defaults=args.extra_args)
        finally:
            # Visualization
            if is_master():
                images_dir = os.path.join(args.save, 'images')
                logger.info(f'images_dir: {images_dir}')
                plot_images(images_dir, args.tensorboard_dir)


def megatron_sft_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronSft(args).main()
