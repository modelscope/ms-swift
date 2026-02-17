# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass
from typing import Literal, Optional

from transformers.utils.versions import require_version

from swift.trainers import Seq2SeqTrainingArguments, TrainArgumentsMixin, TrainerFactory
from swift.utils import (add_version_to_work_dir, get_device_count, get_logger, get_pai_tensorboard_dir, is_master,
                         is_mp, is_pai_training_job, is_swanlab_available, json_parse_to_dict, to_abspath)
from .base_args import BaseArguments
from .tuner_args import TunerArguments

logger = get_logger()


@dataclass
class SwanlabArguments:
    """Arguments for configuring Swanlab for experiment result logging.

    This dataclass stores all the configuration parameters required for initializing and using Swanlab to track
    experiments.

    Args:
        swanlab_token (Optional[str]): The API key for SwanLab. You can also specify it using the `SWANLAB_API_KEY`
            environment variable.
        swanlab_project (str): The SwanLab project, which can be created in advance on the page
            [https://swanlab.cn/space/~](https://swanlab.cn/space/~) or created automatically.
            The default is "ms-swift".
        swanlab_workspace (Optional[str]): The SwanLab workspace. Defaults to `None`, in which case the username
            associated with the API key will be used.
        swanlab_exp_name (Optional[str]): The name of the experiment. If `None`, it will default to the value of the
            `output_dir` argument.
        swanlab_notification_method (Optional[str]): The notification method for SwanLab when training completes
            or errors occur. For details, refer to [here](https://docs.swanlab.cn/plugin/notification-dingtalk.html).
            Supports 'dingtalk', 'lark', 'email', 'discord', 'wxwork', 'slack'.
        swanlab_webhook_url (Optional[str]): Defaults to None. The webhook URL corresponding to
            SwanLab's `swanlab_notification_method`.
        swanlab_secret (Optional[str]): Defaults to None. The secret corresponding to
            SwanLab's `swanlab_notification_method`.
        swanlab_sender_email (Optional[str]): The email address of the sender. Required when
            `swanlab_notification_method` is 'email'.
        swanlab_receiver_email (Optional[str]): The email address of the receiver. Required when
            `swanlab_notification_method` is 'email'.
        swanlab_smtp_server (Optional[str]): The SMTP server address for email notification (e.g., 'smtp.qq.com').
        swanlab_smtp_port (Optional[int]): The SMTP server port for email notification (e.g., 465).
        swanlab_email_language (Optional[str]): email messages language. Supports 'zh', 'en'. The default is "zh".
        swanlab_mode (Literal['cloud', 'local']): The operation mode, either 'cloud' for cloud-based logging or 'local'
            for local-only logging.
    """
    swanlab_token: Optional[str] = None
    swanlab_project: str = 'ms-swift'
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_notification_method: Optional[str] = None
    swanlab_webhook_url: Optional[str] = None
    swanlab_secret: Optional[str] = None
    swanlab_sender_email: Optional[str] = None
    swanlab_receiver_email: Optional[str] = None
    swanlab_smtp_server: Optional[str] = None
    swanlab_smtp_port: Optional[int] = None
    swanlab_email_language: Optional[str] = 'zh'
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

        if self.swanlab_notification_method is not None:
            from swanlab.plugin.notification import (LarkCallback, DingTalkCallback, EmailCallback, DiscordCallback,
                                                     WXWorkCallback, SlackCallback)
            notification_mapping = {
                'lark': LarkCallback,
                'dingtalk': DingTalkCallback,
                'email': EmailCallback,
                'discord': DiscordCallback,
                'wxwork': WXWorkCallback,
                'slack': SlackCallback,
            }
            callback_cls = notification_mapping.get(self.swanlab_notification_method)
            if callback_cls is None:
                raise ValueError(
                    f'Unsupported swanlab_notification_method: "{self.swanlab_notification_method}". Supported methods'
                    f' are: {list(notification_mapping.keys())}')

            if self.swanlab_notification_method == 'email':
                if not (self.swanlab_sender_email and self.swanlab_receiver_email and self.swanlab_smtp_server
                        and self.swanlab_smtp_port):
                    raise ValueError("When 'swanlab_notification_method' is 'email', both 'swanlab_sender_email' "
                                     "and 'swanlab_receiver_email' and 'swanlab_smtp_server' and 'swanlab_smtp_port' "
                                     'must be provided.')
                callback = EmailCallback(
                    sender_email=self.swanlab_sender_email,
                    receiver_email=self.swanlab_receiver_email,
                    password=self.swanlab_secret,
                    smtp_server=self.swanlab_smtp_server,
                    port=self.swanlab_smtp_port,
                    language=self.swanlab_email_language)
            else:
                callback = callback_cls(
                    webhook_url=self.swanlab_webhook_url,
                    secret=self.swanlab_secret,
                )
            swanlab.register_callbacks([callback])

        INTEGRATION_TO_CALLBACK['swanlab'] = SwanLabCallback(
            project=self.swanlab_project,
            workspace=self.swanlab_workspace,
            experiment_name=self.swanlab_exp_name,
            config={'UPPERFRAME': '🐦‍⬛ms-swift'},
            mode=self.swanlab_mode,
        )


@dataclass
class SftArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingArguments):
    """Arguments pertaining to the training process.

    SftArguments is a dataclass that inherits from multiple argument classes: SwanlabArguments, TunerArguments,
    BaseArguments, TrainArgumentsMixin, Seq2SeqTrainingArguments.

    Args:
        add_version (bool): Whether to add a versioned subdirectory like '<version>-<timestamp>' to the `output_dir` to
            prevent overwriting existing checkpoints. Defaults to True.
        create_checkpoint_symlink (bool): Whether to create additional symbolic links for checkpoints, which can be
            useful for automated training scripts. The symlinks for the best and last models will be created at
            `f'{output_dir}/best'` and `f'{output_dir}/last'`, respectively. Defaults to False.
        output_dir (Optional[str]): The directory to save model outputs. Defaults to 'output/<model_name>'.
        learning_rate (Optional[float]): The learning rate. Defaults to 1e-5 for full-parameter training and 1e-4 for
            tuners like LoRA.
            Note: To set a minimum learning rate (min_lr), you can pass the arguments
            --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'.
        eval_strategy (Optional[str]): The evaluation strategy. By default, it aligns with `save_strategy`. It will
            default to 'no' if no validation dataset is provided (i.e., `val_dataset` and `eval_dataset` are not used,
            and `split_dataset_ratio` is 0).
        fp16 (Optional[bool]): Defaults to None.
        bf16 (Optional[bool]): Defaults to None.
        max_new_tokens (int): Overrides generation parameters. The maximum number of new tokens to generate when
            `predict_with_generate` is True. Defaults to 64.
        temperature (float): Overrides generation parameters. The temperature for sampling when `predict_with_generate`
            is True. Defaults to 0.0.
        load_args (bool): Whether to load `args.json` from a saved directory when `--resume_from_checkpoint`,
            `--model`, or `--adapters` is specified. For details on which keys are loaded, refer to `base_args.py`.
            Defaults to `True` for inference and exporting, and `False` for training. This argument typically does not
            need to be modified.
        zero_hpz_partition_size (Optional[int]): A feature of ZeRO++. Enables model sharding within a node and data
            sharding between nodes. If you encounter `grad_norm` NaN issues, consider trying `--torch_dtype float16`.
            Defaults to None.
        deepspeed_autotp_size (Optional[int]): The tensor parallelism size for DeepSpeed AutoTP. To use this, the
            `--deepspeed` argument must be set to 'zero0', 'zero1', or 'zero2'. Note: This feature only supports
            full-parameter fine-tuning. Defaults to None.
    """
    add_version: bool = True
    create_checkpoint_symlink: bool = False

    # override
    output_dir: Optional[str] = None
    learning_rate: Optional[float] = None
    eval_strategy: Optional[str] = None  # steps, epoch
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None

    # extra
    max_new_tokens: int = 64
    temperature: float = 0.
    load_args: bool = False

    # zero++
    zero_hpz_partition_size: Optional[int] = None

    # auto_tp
    deepspeed_autotp_size: Optional[int] = None

    # fsdp
    fsdp: Optional[str] = None

    def _check_padding_free(self):
        if self.padding_free or self.packing:
            if self.packing:
                feature = 'packing'
                self.padding_free = True
            else:
                feature = 'padding_free'
            if self.attn_impl not in {'flash_attn', 'flash_attention_2', 'flash_attention_3'}:
                raise ValueError(f'The "{feature}" feature requires a flash attention implementation. '
                                 'Please use one of: "flash_attn", "flash_attention_2", "flash_attention_3".')

    def __post_init__(self) -> None:
        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)
            # The non-resume_only_model will have its weights loaded in the trainer.
            if self.resume_only_model:
                if self.tuner_type == 'full':
                    self.model = self.resume_from_checkpoint
                else:
                    self.adapters = [self.resume_from_checkpoint]
        BaseArguments.__post_init__(self)
        self._init_override()
        TunerArguments.__post_init__(self)
        self._check_padding_free()
        if self.optimizer is None:
            if self.lorap_lr_ratio:
                self.optimizer = 'lorap'
            elif self.use_galore:
                self.optimizer = 'galore'

        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')

        self._handle_pai_compat()

        self._init_deepspeed()
        self._init_fsdp()
        self._init_device()

        if getattr(self, 'accelerator_config', None) is None:
            self.accelerator_config = {'dispatch_batches': False}
        if not (self.eval_dataset or self._val_dataset_exists):
            self.eval_strategy = 'no'
        self.training_args = TrainerFactory.get_training_args(self)
        self.training_args.remove_unused_columns = False
        self._add_version()

        if 'swanlab' in self.report_to:
            self._init_swanlab()

    def _init_override(self):
        self._init_output_dir()
        self._init_metric()

        if self.learning_rate is None:
            if self.tuner_type == 'full':
                self.learning_rate = 1e-5
            else:
                self.learning_rate = 1e-4
        self._init_eval_strategy()

    def _init_deepspeed(self):
        if self.deepspeed:
            require_version('deepspeed')
            if is_mp() and not self.use_ray:
                raise ValueError('DeepSpeed is not compatible with `device_map`. '
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
            deepspeed_mapping = {
                name: f'{name}.json'
                for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
            }
            for ds_name, ds_config in deepspeed_mapping.items():
                if self.deepspeed == ds_name:
                    self.deepspeed = os.path.join(ds_config_folder, ds_config)
                    break

            self.deepspeed = json_parse_to_dict(self.deepspeed)
            if self.zero_hpz_partition_size is not None:
                assert 'zero_optimization' in self.deepspeed
                self.deepspeed['zero_optimization']['zero_hpz_partition_size'] = self.zero_hpz_partition_size
                logger.warn('If `zero_hpz_partition_size`(ZeRO++) causes grad_norm NaN, please'
                            ' try `--torch_dtype float16`')
            if self.deepspeed_autotp_size is not None:
                assert self.deepspeed is not None, (
                    'To use `deepspeed_autotp_size`, you need to additionally set the `--deepspeed` argument.')
                self.deepspeed['tensor_parallel'] = {'autotp_size': self.deepspeed_autotp_size}
                self.deepspeed['zero_optimization']['gather_16bit_weights_on_model_save'] = True
            logger.info(f'Using deepspeed: {self.deepspeed}')

    def _init_fsdp(self):
        if not self.fsdp:
            self.fsdp = []
            return

        if is_mp() and not self.use_ray:
            raise ValueError('FSDP2 is not compatible with `device_map`. '
                             f'n_gpu: {get_device_count()}, '
                             f'local_world_size: {self.local_world_size}.')
        if self.deepspeed:
            raise ValueError('FSDP2 is not compatible with DeepSpeed.')

        fsdp_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

        # FSDP2 preset configurations
        fsdp_mapping = {
            'fsdp2': 'fsdp2.json',
        }

        fsdp_config_path = self.fsdp
        for fsdp_name, fsdp_config in fsdp_mapping.items():
            if self.fsdp == fsdp_name:
                fsdp_config_path = os.path.join(fsdp_config_folder, fsdp_config)
                break

        fsdp_config_dict = json_parse_to_dict(fsdp_config_path)

        # Extract fsdp string options (e.g., "full_shard auto_wrap offload")
        fsdp_options = fsdp_config_dict.get('fsdp', 'full_shard auto_wrap')
        self.fsdp = fsdp_options

        # Extract fsdp_config dict
        self.fsdp_config = fsdp_config_dict.get('fsdp_config', {})

        # Set FSDP_VERSION environment variable for accelerate to recognize FSDP2
        fsdp_version = self.fsdp_config.get('fsdp_version', 2)
        os.environ['FSDP_VERSION'] = str(fsdp_version)

        # Set environment variable to optimize NCCL memory usage
        if 'TORCH_NCCL_AVOID_RECORD_STREAMS' not in os.environ:
            os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'

        # Check FSDP2 compatibility with other training arguments
        self._check_fsdp2_compatibility()

        logger.info(f'Using FSDP2: fsdp={self.fsdp}, fsdp_config={self.fsdp_config}')

    def _check_fsdp2_compatibility(self):
        """Check for incompatible argument combinations with FSDP2.

        FSDP2 has several known limitations:
        1. save_only_model=True + SHARDED_STATE_DICT: Can't save only model weights with sharded state dict
        2. gradient_checkpointing=True: Should use activation_checkpointing in fsdp_config instead
        """
        state_dict_type = self.fsdp_config.get('state_dict_type', 'SHARDED_STATE_DICT')

        # Check 1: save_only_model + SHARDED_STATE_DICT
        if getattr(self, 'save_only_model', False) and 'SHARDED' in state_dict_type.upper():
            raise ValueError(
                'FSDP2 with SHARDED_STATE_DICT is not compatible with save_only_model=True. '
                'Either set save_only_model=False, or change state_dict_type to FULL_STATE_DICT in fsdp_config. '
                'Note: FULL_STATE_DICT requires more memory and is slower.')

        # Check 2: gradient_checkpointing should be disabled, use activation_checkpointing instead
        if getattr(self, 'gradient_checkpointing', False):
            activation_checkpointing = self.fsdp_config.get('activation_checkpointing', False)
            if activation_checkpointing:
                logger.warning('Both gradient_checkpointing and fsdp_config.activation_checkpointing are enabled. '
                               'For FSDP2, it is recommended to use only activation_checkpointing in fsdp_config. '
                               'Disabling gradient_checkpointing automatically.')
                self.gradient_checkpointing = False
            else:
                logger.warning(
                    'gradient_checkpointing is enabled with FSDP2. '
                    'For better performance, consider using activation_checkpointing in fsdp_config instead. '
                    'Add "activation_checkpointing": true to your fsdp_config.')

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

        if self.run_name is None:
            self.run_name = self.output_dir

        self.training_args.output_dir = self.output_dir
        self.training_args.run_name = self.run_name
        self.training_args.logging_dir = self.logging_dir

    def _init_output_dir(self):
        if self.output_dir is None:
            self.output_dir = f'output/{self.model_suffix}'
        self.output_dir = to_abspath(self.output_dir)

    def _init_eval_strategy(self):
        if self.eval_strategy is None:
            self.eval_strategy = self.save_strategy
        if self.eval_strategy == 'no':
            self.eval_steps = None
            if self.split_dataset_ratio > 0:
                self.split_dataset_ratio = 0.
                logger.info(f'Setting args.split_dataset_ratio: {self.split_dataset_ratio}')
        elif self.eval_strategy == 'steps' and self.eval_steps is None:
            self.eval_steps = self.save_steps
        self.evaluation_strategy = self.eval_strategy

    def _init_metric(self):
        if self.eval_metric is None:
            if self.task_type == 'causal_lm' and self.predict_with_generate:
                self.eval_metric = 'nlg'
            elif self.task_type == 'embedding':
                self.eval_metric = 'infonce' if self.loss_type == 'infonce' else 'paired'
            elif self.task_type in {'reranker', 'generative_reranker'}:
                self.eval_metric = 'reranker'
        if self.metric_for_best_model is None:
            self.metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = 'loss' not in self.metric_for_best_model
