# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import torch
from transformers import Seq2SeqTrainingArguments
from transformers.utils.versions import require_version

from swift.plugin import LOSS_MAPPING
from swift.trainers import IntervalStrategy, TrainerFactory
from swift.utils import (add_version_to_work_dir, get_logger, get_pai_tensorboard_dir, is_liger_available,
                         is_local_master, is_mp, is_pai_training_job, use_torchacc)
from .base_args import BaseArguments, to_abspath
from .tuner_args import TunerArguments

logger = get_logger()


@dataclass
class Seq2SeqTrainingOverrideArguments(Seq2SeqTrainingArguments):
    """Override the default value in `Seq2SeqTrainingArguments`"""
    output_dir: Optional[str] = None
    gradient_checkpointing: bool = True

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    logging_steps: int = 5
    learning_rate: Optional[float] = None
    weight_decay: float = 0.1
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    remove_unused_columns: bool = False
    logging_first_step: bool = True
    eval_strategy: Optional[str] = None  # steps, epoch

    def _init_output_dir(self):
        if self.output_dir is not None:
            return
        self.output_dir = f'output/{self.model_suffix}'

    def _init_eval_strategy(self):
        if self.eval_strategy is None:
            self.eval_strategy = self.save_strategy
        if self.eval_strategy == 'no':
            self.eval_steps = None
            self.split_dataset_ratio = 0.
            logger.info(f'Setting args.split_dataset_ratio: {self.split_dataset_ratio}')
        elif self.eval_strategy == 'steps' and self.eval_steps is None:
            self.eval_steps = self.save_steps
        self.evaluation_strategy = self.eval_strategy

    def __post_init__(self):
        self._init_output_dir()
        if self.metric_for_best_model is None:
            self.metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'
        if self.greater_is_better is None:
            self.greater_is_better = 'loss' not in self.metric_for_best_model

        if self.learning_rate is None:
            if self.train_type == 'full':
                self.learning_rate = 1e-5
            else:
                self.learning_rate = 1e-4
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = self.parse_to_dict(self.lr_scheduler_kwargs)
        if getattr(self, 'gradient_checkpointing_kwargs', None):
            self.gradient_checkpointing_kwargs = self.parse_to_dict(self.gradient_checkpointing_kwargs)
        self._init_eval_strategy()


@dataclass
class TorchAccArguments:
    model_layer_cls_name: Optional[str] = field(
        default=None,
        metadata={'help': "Decoder Class name of model, e.g. 'QWenBlock' for QWen, 'LlamaDecoderLayer' for LLama"})
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1
    acc_steps: int = 1

    def __post_init__(self):
        """Prepare torchacc"""
        if use_torchacc():
            self.dataloader_drop_last = True


@dataclass
class TrainArguments(TorchAccArguments, TunerArguments, Seq2SeqTrainingOverrideArguments, BaseArguments):
    """
    TrainArguments class is a dataclass that inherits from multiple argument classes:
    TorchAccArguments, TunerArguments, Seq2SeqTrainingOverrideArguments, and BaseArguments.

    Args:
        add_version (bool): Flag to add version information to output_dir. Default is True.
        resume_only_model (bool): Flag to resume training only the model. Default is False.
        check_model (bool): Flag to check the model is latest. Default is True.
        loss_type (Optional[str]): Type of loss function to use. Default is None.
        num_labels (Optional[int]): Number of labels for classification tasks. Default is None.
        packing (bool): Flag to enable packing of datasets. Default is False.
        lazy_tokenize (Optional[bool]): Flag to enable lazy tokenization. Default is None.
        acc_strategy (Literal['token', 'seq']): Strategy for accumulation. Default is 'token'.
        max_new_tokens (int): Maximum number of new tokens to generate. Default is 64.
        temperature (float): Temperature for sampling. Default is 0.
        optimizer (Optional[str]): Optimizer type to use, define it in the plugin package. Default is None.
        metric (Optional[str]): Metric to use for evaluation, define it in the plugin package. Default is None.
    """
    add_version: bool = True
    resume_only_model: bool = False
    check_model: bool = True
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})

    # dataset
    packing: bool = False
    lazy_tokenize: Optional[bool] = None

    # extra
    acc_strategy: Literal['token', 'seq'] = 'token'
    max_new_tokens: int = 64
    temperature: float = 0.
    optimizer: Optional[str] = None
    metric: Optional[str] = None

    def __post_init__(self) -> None:
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)
            if self.train_type == 'full':
                self.model = self.resume_from_checkpoint
            else:
                self.adapters = [self.resume_from_checkpoint]
        BaseArguments.__post_init__(self)
        Seq2SeqTrainingOverrideArguments.__post_init__(self)
        TunerArguments.__post_init__(self)
        TorchAccArguments.__post_init__(self)

        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        self._handle_pai_compat()
        self._init_liger()

        self._init_deepspeed()
        self._init_device()

        if self.streaming and self.lazy_tokenize:
            self.lazy_tokenize = False
            logger.warning('Streaming and lazy_tokenize are incompatible. '
                           f'Setting args.lazy_tokenize: {self.lazy_tokenize}.')
        if self.lazy_tokenize is None:
            self.lazy_tokenize = self.model_meta.is_multimodal and not self.streaming
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        if getattr(self, 'accelerator_config', None) is None:
            self.accelerator_config = {'dispatch_batches': False}
        self.training_args = TrainerFactory.get_training_args(self)

        self._add_version()

    def _init_deepspeed(self):
        if self.deepspeed:
            require_version('deepspeed')
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))
            deepspeed_mapping = {name: f'{name}.json' for name in ['zero2', 'zero3', 'zero2_offload', 'zero3_offload']}
            for ds_name, ds_config in deepspeed_mapping.items():
                if self.deepspeed == ds_name:
                    self.deepspeed = os.path.join(ds_config_folder, ds_config)
                    break

            self.deepspeed = self.parse_to_dict(self.deepspeed)
            logger.info(f'Using deepspeed: {self.deepspeed}')

    def _init_liger(self):
        if self.use_liger:
            assert is_liger_available(), 'use_liger requires liger_kernels, try `pip install liger-kernel`'
            if self.loss_scale != 'default':
                logger.warning('use_liger is not compatible with `loss_scale`, setting to default...')
                self.loss_scale = 'default'

    def _handle_pai_compat(self) -> None:
        if not is_pai_training_job():
            return

        logger.info('Handle pai compat...')
        pai_tensorboard_dir = get_pai_tensorboard_dir()
        if self.logging_dir is None and pai_tensorboard_dir is not None:
            self.logging_dir = pai_tensorboard_dir
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')
        self.add_version = False
        logger.info(f'Setting args.add_version: {self.add_version}')

    def _add_version(self):
        """Prepare the output_dir"""
        if self.add_version:
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'output_dir: {self.output_dir}')

        if self.logging_dir is None:
            self.logging_dir = f'{self.output_dir}/runs'

        self.output_dir = to_abspath(self.output_dir)
        self.logging_dir = to_abspath(self.logging_dir)
        if is_local_master():
            os.makedirs(self.output_dir, exist_ok=True)

        self.training_args.output_dir = self.output_dir
        self.training_args.run_name = self.output_dir
        self.training_args.logging_dir = self.logging_dir
