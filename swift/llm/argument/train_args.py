# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.llm import MODEL_ARCH_MAPPING, MODEL_MAPPING
from swift.plugin import LOSS_MAPPING, extra_tuners
from swift.trainers import IntervalStrategy, TrainerFactory
from swift.utils import (add_version_to_work_dir, get_dist_setting, get_logger, get_pai_tensorboard_dir, is_dist,
                         is_liger_available, is_local_master, is_mp, is_pai_training_job, use_torchacc)
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
    lr_scheduler_kwargs: Optional[str] = None  # json
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    remove_unused_columns: bool = False
    logging_first_step: bool = True

    def _init_output_dir(self):
        if self.output_dir is not None:
            return
        self.output_dir = f'output/{self.model_name}'

    def __post_init__(self):
        del TrainingArguments.world_size
        self._init_output_dir()

        if self.learning_rate is None:
            if self.train_type == 'full':
                self.learning_rate = 1e-5
            else:
                self.learning_rate = 1e-4
        self.lr_scheduler_kwargs = self.parse_to_dict(self.lr_scheduler_kwargs)
        self.gradient_checkpointing_kwargs = self.parse_to_dict(self.gradient_checkpointing_kwargs)

        if len(self.val_dataset) == 0 and self.split_dataset_ratio == 0:
            self.evaluation_strategy = IntervalStrategy.NO
            self.eval_strategy = IntervalStrategy.NO
            self.eval_steps = None
        elif self.eval_steps is None:
            self.evaluation_strategy = self.save_strategy
            self.eval_strategy = self.save_strategy
            self.eval_steps = self.save_steps


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
    TrainArguments class is a dataclass that holds various arguments related to training configuration and usage.

    Args:
        freeze_vit (bool): Flag to indicate if ViT should be frozen. Default is True.
        freeze_aligner (bool): Flag to indicate if aligner should be frozen. Default is True.
        freeze_llm (bool): Flag to indicate if LLM should be frozen. Default is False.
        freeze_parameters (List[str]): List of parameters to freeze. Default is an empty list.
        freeze_parameters_ratio (float): Ratio of parameters to freeze. Default is 0.
        additional_trainable_parameters (List[str]): List of additional trainable parameters. Default is an empty list.
        add_version (bool): Flag to indicate if output directory suffix should be added. Default is True.
        resume_from_checkpoint (Optional[str]): Path to resume from checkpoint. Default is None.
        resume_only_model (bool): Flag to indicate if only the model should be resumed when resume-training.
            Default is False.
        check_model (bool): Flag to check if the model is the latest. Default is True.
            Turn this to False if you network is unstable.
        loss_type (Optional[str]): Type of loss function. Default is None.
        packing (bool): Flag to indicate if packing is used. Default is False.
        lazy_tokenize (Optional[bool]): Flag to indicate if lazy tokenization is used. Default is None.
        acc_strategy (Literal): Strategy for accuracy calculation. Default is 'token'.
    """
    add_version: bool = True
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    check_model: bool = True
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})

    # dataset
    packing: bool = False
    lazy_tokenize: Optional[bool] = None

    # extra
    acc_strategy: Literal['token', 'sentence'] = 'token'

    def __post_init__(self) -> None:
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)
            self.load_args_from_ckpt(self.resume_from_checkpoint)
            if self.train_type == 'full':
                self.model_id_or_path = self.resume_from_checkpoint
        Seq2SeqTrainingOverrideArguments.__post_init__(self)
        BaseArguments.__post_init__(self)
        TunerArguments.__post_init__(self)
        TorchAccArguments.__post_init__(self)

        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        self._handle_pai_compat()
        self._init_liger()

        self._init_deepspeed()
        self._init_device()
        self.seed += self.rank  # Avoid the same dropout

        if self.lazy_tokenize is None:
            self.lazy_tokenize = self.model_meta.is_multimodal and not self.streaming
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        self.accelerator_config = {'dispatch_batches': False}
        self.training_args = TrainerFactory.get_training_args(self)

        self._add_version()
        self.save_args()

    def _init_deepspeed(self):
        """Prepare deepspeed settings"""
        ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))
        deepspeed_mapping = {name: f'{name}.json' for name in ['zero2', 'zero3', 'zero2_offload', 'zero3_offload']}
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.deepspeed == ds_name:
                self.deepspeed = os.path.join(ds_config_folder, ds_config)
                break

        if self.deepspeed is not None:
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with MP. '
                                 f'n_gpu: {torch.cuda.device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')
            require_version('deepspeed')
            self.parse_to_dict(self.deepspeed)
            logger.info(f'Using deepspeed: {self.deepspeed}')

    def _init_liger(self):
        """Liger kernel"""
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
        """Prepare the output folder"""
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
