# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import Seq2SeqTrainingArguments
from transformers.utils.versions import require_version

from swift.plugin import LOSS_MAPPING
from swift.trainers import TrainerFactory
from swift.trainers.arguments import TrainArgumentsMixin
from swift.utils import (add_version_to_work_dir, get_device_count, get_logger, get_pai_tensorboard_dir, is_master,
                         is_mp, is_pai_training_job, is_swanlab_available)
from .base_args import BaseArguments, to_abspath
from .tuner_args import TunerArguments

logger = get_logger()


@dataclass
class Seq2SeqTrainingOverrideArguments(TrainArgumentsMixin, Seq2SeqTrainingArguments):
    """Override the default value in `Seq2SeqTrainingArguments`"""
    output_dir: Optional[str] = None
    learning_rate: Optional[float] = None
    eval_strategy: Optional[str] = None  # steps, epoch

    def _init_output_dir(self):
        if self.output_dir is None:
            self.output_dir = f'output/{self.model_suffix}'
        self.output_dir = to_abspath(self.output_dir)

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

    def _init_metric_for_best_model(self):
        if self.metric_for_best_model is None:
            self.metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'

    def __post_init__(self):
        self._init_output_dir()
        self._init_metric_for_best_model()
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = 'loss' not in self.metric_for_best_model

        if self.learning_rate is None:
            if self.train_type == 'full':
                self.learning_rate = 1e-5
            else:
                self.learning_rate = 1e-4
        self._init_eval_strategy()


@dataclass
class SwanlabArguments:

    swanlab_token: Optional[str] = None
    swanlab_project: Optional[str] = None
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_mode: Literal['cloud', 'local'] = 'cloud'

    def _init_swanlab(self):
        if not is_swanlab_available():
            raise ValueError('You are using swanlab as `report_to`, please install swanlab by ' '`pip install swanlab`')
        if not self.swanlab_exp_name:
            self.swanlab_exp_name = self.output_dir
        from transformers.integrations import INTEGRATION_TO_CALLBACK
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
        if self.swanlab_token:
            swanlab.login(self.swanlab_token)
        INTEGRATION_TO_CALLBACK['swanlab'] = SwanLabCallback(
            project=self.swanlab_project,
            workspace=self.swanlab_workspace,
            experiment_name=self.swanlab_exp_name,
            config={'UPPERFRAME': '🐦‍⬛ms-swift'},
            mode=self.swanlab_mode,
        )


@dataclass
class TrainArguments(SwanlabArguments, TunerArguments, Seq2SeqTrainingOverrideArguments, BaseArguments):
    """
    TrainArguments class is a dataclass that inherits from multiple argument classes:
    TunerArguments, Seq2SeqTrainingOverrideArguments, and BaseArguments.

    Args:
        add_version (bool): Flag to add version information to output_dir. Default is True.
        resume_only_model (bool): Flag to resume training only the model. Default is False.
        loss_type (Optional[str]): Type of loss function to use. Default is None.
        packing (bool): Flag to enable packing of datasets. Default is False.
        lazy_tokenize (Optional[bool]): Flag to enable lazy tokenization. Default is None.
        max_new_tokens (int): Maximum number of new tokens to generate. Default is 64.
        temperature (float): Temperature for sampling. Default is 0.
        optimizer (Optional[str]): Optimizer type to use, define it in the plugin package. Default is None.
        metric (Optional[str]): Metric to use for evaluation, define it in the plugin package. Default is None.
    """
    add_version: bool = True
    resume_only_model: bool = False
    create_checkpoint_symlink: bool = False

    # dataset
    packing: bool = False
    lazy_tokenize: Optional[bool] = None

    # plugin
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})
    optimizer: Optional[str] = None
    metric: Optional[str] = None

    # extra
    max_new_tokens: int = 64
    temperature: float = 0.
    load_args: bool = False

    # zero++
    zero_hpz_partition_size: Optional[int] = None

    def _init_lazy_tokenize(self):
        if self.streaming and self.lazy_tokenize:
            self.lazy_tokenize = False
            logger.warning('Streaming and lazy_tokenize are incompatible. '
                           f'Setting args.lazy_tokenize: {self.lazy_tokenize}.')
        if self.lazy_tokenize is None:
            self.lazy_tokenize = self.model_meta.is_multimodal and not self.streaming
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')

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

        if self.optimizer is None:
            if self.lorap_lr_ratio:
                self.optimizer = 'lorap'
            elif self.use_galore:
                self.optimizer = 'galore'

        if len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the training dataset.')

        self._handle_pai_compat()

        self._init_deepspeed()
        self._init_device()
        self._init_lazy_tokenize()

        if getattr(self, 'accelerator_config', None) is None:
            self.accelerator_config = {'dispatch_batches': False}
        self.training_args = TrainerFactory.get_training_args(self)
        self.training_args.remove_unused_columns = False

        self._add_version()

        if 'swanlab' in self.report_to:
            self._init_swanlab()

    def _init_deepspeed(self):
        if self.deepspeed:
            require_version('deepspeed')
            if is_mp():
                raise ValueError('DeepSpeed is not compatible with `device_map`. '
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))
            deepspeed_mapping = {
                name: f'{name}.json'
                for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
            }
            for ds_name, ds_config in deepspeed_mapping.items():
                if self.deepspeed == ds_name:
                    self.deepspeed = os.path.join(ds_config_folder, ds_config)
                    break

            self.deepspeed = self.parse_to_dict(self.deepspeed)
            if self.zero_hpz_partition_size is not None:
                assert 'zero_optimization' in self.deepspeed
                self.deepspeed['zero_optimization']['zero_hpz_partition_size'] = self.zero_hpz_partition_size
                logger.warn('If `zero_hpz_partition_size`(ZeRO++) causes grad_norm NaN, please'
                            ' try `--torch_dtype float16`')
            logger.info(f'Using deepspeed: {self.deepspeed}')

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

        self.logging_dir = to_abspath(self.logging_dir)
        if is_master():
            os.makedirs(self.output_dir, exist_ok=True)

        self.training_args.output_dir = self.output_dir
        self.training_args.run_name = self.output_dir
        self.training_args.logging_dir = self.logging_dir
