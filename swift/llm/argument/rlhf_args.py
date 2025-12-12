# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.llm import MODEL_MAPPING
from swift.trainers import GRPOArgumentsMixin, RLHFArgumentsMixin
from swift.utils import get_current_device, get_logger, is_master, is_mp, json_parse_to_dict, set_default_ddp_config
from .train_args import TrainArguments

logger = get_logger()
rlhf_support_vllm_types = ['grpo', 'gkd']


@dataclass
class RewardModelArguments:
    """Arguments pertaining to the reward model.

    Args:
        reward_model (Optional[List[str]]): The model ID or a local path to the reward model. Same as the `model`
            argument. Defaults to None.
        reward_adapters (List[str]): The path(s) to LoRA adapter weights to be loaded for the reward model. Useful for
            using LoRA weights from SFT as the reward model. Defaults to an empty list (`[]`).
        reward_model_type (Optional[List[str]]): The model type of the reward model. Same as the `model_type` argument.
            If not specified, it's often inferred. Defaults to None.
        reward_model_revision (Optional[List[str]]): The specific model version to use for the reward model. Same as
            the `model_revision` argument. Defaults to None.
    """
    reward_model: Optional[List[str]] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[List[str]] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[List[str]] = None


@dataclass
class TeacherModelArguments:
    """Arguments for configuring the teacher model.

    Args:
        teacher_model (Optional[str]): The model ID or a local path to the teacher model. This is required when
            `rlhf_type` is 'gkd'. Analogous to the main `model` argument. Defaults to None.
        teacher_adapters (List[str]): A list of paths to LoRA weights. These weights, often produced by SFT, are loaded
            to form the teacher model. Defaults to an empty list (`[]`).
        teacher_model_type (Optional[str]): The model type of the teacher model. If not specified, it's often inferred.
            Analogous to the main `model_type` argument. Defaults to None.
        teacher_model_revision (Optional[str]): The specific model version of the teacher model to use. Analogous to
            the main `model_revision` argument. Defaults to None.
        teacher_deepspeed (Optional[str]): The teacher model's deepspeed configuration. This can be a JSON file path or
            one of the following values: 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'. If not
            provided, it defaults to using the same DeepSpeed configuration as the main training model. Analogous to
            the main `deepspeed` argument.
    """
    teacher_model: Optional[str] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    teacher_model_revision: Optional[str] = None
    teacher_deepspeed: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'DeepSpeed configuration for teacher model. '
            'Can be a path to a json file or one of: zero0, zero1, zero2, zero3, zero2_offload, zero3_offload'
        })


@dataclass
class PPOArguments:
    """Arguments for configuring the PPO training.

    Args:
        num_ppo_epochs (int): Number of epochs to train. Defaults to 4.
        whiten_rewards (bool): Whether to whiten the rewards. Defaults to False.
        kl_coef (float): KL coefficient. Defaults to 0.05.
        cliprange (float): Clip range. Defaults to 0.2.
        vf_coef (float): Value function coefficient. Defaults to 0.1.
        cliprange_value (float): Clip range for the value function. Defaults to 0.2.
        gamma (float): Discount factor. Defaults to 1.0.
        lam (float): Lambda value for GAE. Defaults to 0.95.
        num_mini_batches (int): Defaults to 1.
        local_rollout_forward_batch_size (int): Defaults to 64.
        num_sample_generations (int): Number of generations. Defaults to 10.
        response_length (Optional[int]): (Deprecated) Compatibility parameter. Use `max_completion_length` instead.
            Defaults to None.
        missing_eos_penalty (Optional[float]): Defaults to None.
    """
    num_ppo_epochs: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    cliprange: float = 0.2
    vf_coef: float = 0.1
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95

    num_mini_batches: int = 1
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    response_length: Optional[int] = None  # compat. use max_completion_length instead
    missing_eos_penalty: Optional[float] = None


@dataclass
class GRPOArguments(GRPOArgumentsMixin):
    """A dataclass for configuring GRPO training.

    These arguments control the hyperparameters specific to the GRPO algorithm.

    Args:
        num_generations  (int): The number of completions to generate for each prompt. This corresponds to the G value
            in the GRPO paper. The total generation batch size (e.g., `generation_batch_size` or `steps_per_generation
            * per_device_batch_size * num_processes`) must be divisible by this number. Defaults to 8.
        reward_funcs (List[str]): A list of reward function names to use for the GRPO algorithm. Available built-in
            options include 'accuracy', 'format', 'cosine', 'repetition', and 'soft_overlong'
            (see swift/plugin/orm.py). Custom reward functions can also be defined. Defaults to an empty list.
        reward_weights (List[float]): A list of weights for each reward source. The length must match the total number
            of reward functions (from `reward_funcs`) plus any external reward models. If `None`, all rewards are
            weighted equally with a value of 1.0. Note: If an external `--reward_model` is used, it is treated as the
            last reward source in the sequence. Defaults to None.
        log_completions (bool): Whether to log the model's generated completions during training. This is designed to
            be used with an experiment tracker like WandB or SwanLab (`--report_to wandb`/`swanlab`). If enabled
            without a tracker, completions are saved to `completions.jsonl` in the checkpoint directory. Defaults to
            False.
        num_iterations (int): The number of update steps to perform for each data sample. This corresponds to the K
            value in the GRPO paper. Defaults to 1.
        truncation_strategy (Literal['delete', 'left', 'right', 'split', None]): The strategy for handling input
            sequences that exceed `max_length`. Supported options: 'delete' to discard the sample, 'left' to truncate
            from the beginning, 'right' to truncate from the end. Defaults to None, and then sets to 'left' in the
            `_init_grpo` function.
            Note that for multimodal models, left pruning may prune multimodal tokens, causing shape mismatch errors
            in the forward feed. Using the `delete` method will resample other data from the original dataset to
            supplement excessively long data and examples with encoding failures.
    """
    num_generations: int = 8  # G in the GRPO paper
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    log_completions: bool = False

    # multi step
    num_iterations: int = 1

    truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None


@dataclass
class RLHFArguments(TeacherModelArguments, GRPOArguments, PPOArguments, RewardModelArguments, RLHFArgumentsMixin,
                    TrainArguments):
    """A dataclass holding arguments for Reinforcement Learning from Human Feedback.

    Args:
        rlhf_type (str): The type of human alignment algorithm to use. Supports 'dpo', 'orpo', 'simpo', 'kto', 'cpo',
            'rm', 'ppo', 'grpo', and 'gkd'. Defaults to 'dpo'.
        ref_model (Optional[str]): The model path for the reference model. Required when using 'dpo', 'kto', 'ppo',
            or 'grpo' with full-parameter training. Defaults to None, which will set it to the value of the `--model`
            argument.
        ref_adapters (List[str]): LoRA adapters for the reference model. If you are using LoRA weights from SFT for
            DPO/KTO/GRPO, set both `--adapters` and `--ref_adapters` to the SFT checkpoint path. When resuming from an
            RLHF checkpoint, set `--resume_from_checkpoint` to the RLHF checkpoint and `--ref_adapters` to the SFT
            checkpoint. Defaults to an empty list.
        ref_model_type (Optional[str]): The model type of the reference model. Same as `model_type`. Defaults to None.
        ref_model_revision (Optional[str]): The model revision of the reference model. Same as `model_revision`.
            Defaults to None.
        beta (Optional[float]): The beta parameter for RLHF, controlling the deviation from the reference model.
            A higher value implies less deviation. If None, uses algorithm-specific defaults: 2.0 for 'simpo', 0.04
            for 'grpo', 0.5 for 'gkd', and 0.1 for others. Defaults to None.
        label_smoothing (float): The label smoothing value for DPO. A value of 0 disables it. Defaults to 0.
        max_completion_length (int): The maximum generation length for GRPO/PPO/GKD algorithms. Defaults to 512.
        loss_scale (Optional[str]): Overrides the template parameter. During RLHF training, this defaults to
            'last_round'.
        rpo_alpha (Optional[float]): The alpha parameter from the RPO paper, controlling the weight of the SFT loss
            (NLL term). The loss is calculated as `dpo_loss + rpo_alpha * sft_loss`. If None, the SFT loss is not
            included. Note: The default was 1.0 in `ms-swift<3.8` and changed to None in `ms-swift>=3.8`. Defaults to
            None.
        ld_alpha (Optional[float]): The alpha parameter from the LD-DPO paper, which weights the log probabilities of
            the sequence part beyond the common prefix to mitigate length preference. Defaults to None.
        discopop_tau (float): The temperature parameter from the DiscoPOP paper, used to scale the log-ratio. Effective
            when `loss_type` is 'discopop'. Defaults to 0.05.
        loss_type (Optional[List[str]]): The type of loss function. Defaults to algorithm-specific values (e.g.,
            'sigmoid' for DPO). Multiple values can be passed for mixed training (MPO), which requires `loss_weights`
            to be set.
        loss_weights (Optional[List[float]]): When multiple `loss_type` values are set for DPO, this specifies the
            weights for each loss term. Defaults to None.
        cpo_alpha (float): The coefficient for the NLL loss in the CPO/SimPO loss function. Defaults to 1.0.
        simpo_gamma (float): The reward margin term in the SimPO algorithm. The paper suggests a value between 0.5 and
            1.5. Defaults to 1.0.
        desirable_weight (float): In KTO, the weight applied to the desirable loss to counteract data imbalance.
            Defaults to 1.0.
        undesirable_weight (float): In KTO, the weight applied to the undesirable loss to counteract data imbalance.
            Defaults to 1.0.
        temperature (float): The temperature for sampling, used in PPO, GRPO, and GKD algorithms. Defaults to 0.9.
        center_rewards_coefficient (Optional[float]): Used for Reward Model (RM) training. A coefficient to encourage
            the reward model to output rewards with a mean of zero. A value of 0.01 is recommended. Defaults to None.
        lmbda (float): The lambda parameter for GKD, balancing policy and value losses. Defaults to 0.5.
        seq_kd (bool): Whether to use sequence-level knowledge distillation for GKD. Defaults to False.
        offload_teacher_model (bool): Whether to offload the teacher model to CPU memory to save VRAM during GKD
            training. Defaults to False.
        max_new_tokens (Optional[int]): A backward-compatibility argument. Please use `max_completion_length` instead.
            Defaults to None.
    """
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'] = 'dpo'
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
    max_completion_length: int = 512
    loss_scale: Optional[str] = None  # 'last_round'
    # DPO
    rpo_alpha: Optional[float] = None
    ld_alpha: Optional[float] = None  # α parameter from the LD-DPO paper
    discopop_tau: float = 0.05  # τ/temperature parameter from the DiscoPOP paper
    loss_type: Optional[List[str]] = None
    loss_weights: Optional[List[float]] = None
    # CPO
    cpo_alpha: float = 1.
    # SimPO
    simpo_gamma: float = 1
    # KTO
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0
    # PPO/GRPO/GKD
    temperature: float = 0.9
    # RM
    center_rewards_coefficient: Optional[float] = None
    # GKD
    lmbda: float = 0.5
    seq_kd: bool = False
    offload_teacher_model: bool = False
    # compat
    max_new_tokens: Optional[int] = None  # use max_completion_length instead

    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:
        if self.rlhf_type == 'ppo':
            training_args['world_size'] = self.global_world_size

    def __post_init__(self):
        self._process_loss_type()
        self._init_grpo()
        self._init_rm()
        self._init_simpo()
        self._init_max_completion_length()
        self._init_padding_side()
        self._set_default()
        self._init_rollout()
        self._init_teacher_deepspeed()
        GRPOArguments.__post_init__(self)
        TrainArguments.__post_init__(self)
        self._check_sequence_parallel()
        self._check_grpo()
        self._check_gkd()

        if self.loss_scale is None:
            if self.rlhf_type == 'orpo' and not self.model_meta.is_multimodal:
                # Avoid padding labels during the model's forward pass in multimodal models.
                # Some multimodal models do not expand the image pad token.
                self.loss_scale = 'default'
            elif self.rlhf_type == 'grpo':
                if self.loss_scale is None:
                    if self.multi_turn_scheduler:
                        self.loss_scale = 'default'
                    else:
                        self.loss_scale = 'last_round'
            else:
                self.loss_scale = 'last_round'
        if isinstance(self.ref_adapters, str):
            self.ref_adapters = [self.ref_adapters]
        if self.rlhf_type == 'grpo' and self.beta == 0.0:
            self.ref_model = None
        elif self.rlhf_type in ['dpo', 'kto', 'ppo', 'grpo'] and self.train_type == 'full':
            self.ref_model = self.ref_model or self.model
            self.ref_model_type = self.ref_model_type or self.model_type
            self.ref_model_revision = self.ref_model_revision or self.model_revision
        elif self.ref_model is not None:
            raise ValueError('CPO/ORPO or LoRA training does not require a ref_model to be passed in.')

    def _process_loss_type(self):
        if self.loss_type is None:
            return

        if isinstance(self.loss_type, list):
            num_loss_types = len(self.loss_type)
            if num_loss_types > 1:
                assert self.rlhf_type == 'dpo', (f'Multiple loss types ({self.loss_type}) are only supported for DPO. '
                                                 f'Current rlhf_type: {self.rlhf_type}.')
                from trl.trainer.dpo_config import DPOConfig
                assert 'loss_weights' in DPOConfig.__dict__, (
                    'Multiple loss types requires trl >= 0.20, please install trl `pip install -U trl`')

        if hasattr(self.loss_type, '__len__') and len(self.loss_type) == 1:
            self.loss_type = self.loss_type[0]

        # Validate loss_type
        if self.loss_weights is not None:
            assert self.rlhf_type == 'dpo'
            loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]
            if len(self.loss_weights) != len(loss_types):
                raise ValueError(f'Length of loss_weights list ({self.loss_weights}) must match number of loss types '
                                 f'({loss_types}).')

    def _init_grpo(self):
        if self.rlhf_type != 'grpo':
            return
        if self.cached_dataset or self.cached_val_dataset:
            raise ValueError('cached_dataset is not supported for GRPO.')
        if self.use_vllm:
            set_default_ddp_config()
        if self.async_generate or not self.use_vllm or self.vllm_mode == 'server':
            self.sleep_level = 0
        self.remove_unused_columns = False
        logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'left'
        assert self.truncation_strategy in ['left', 'delete'
                                            ], ("GRPO requires `truncation_strategy 'left' or 'delete'`, "
                                                f"Current value: `truncation_strategy='{self.truncation_strategy}'`.")
        if self.beta is None:
            self.beta = 0.04  # https://arxiv.org/abs/2402.03300
        if self.async_generate:
            logger.info('Using async mode. This is a approximate version which '
                        'will use the old weights to generate responses to accelerate. '
                        'This will ignore the `CLIP` of advantages, if you found the training '
                        'is unstable, you may consider using --async_generate false.')
        if 'soft_overlong' in self.reward_funcs:
            assert self.soft_cache_length is not None, \
                'The soft_cache_length must be set when using soft overlong rewards.'
            if self.soft_max_length is None:
                self.soft_max_length = self.max_completion_length
                logger.info(f'Auto-configured soft_max_length = max_completion_length {self.max_completion_length}')

        if self.kl_in_reward is None:
            if self.advantage_estimator == 'grpo':
                self.kl_in_reward = False
            elif self.advantage_estimator in ['rloo', 'reinforce_plus_plus']:
                self.kl_in_reward = True
            else:
                raise ValueError(f'Invalid advantage_estimator: {self.advantage_estimator}')

        if self.scale_rewards is None:
            if self.advantage_estimator == 'grpo':
                self.scale_rewards = 'group'
            elif self.advantage_estimator == 'rloo':
                self.scale_rewards = 'none'
            elif self.advantage_estimator == 'reinforce_plus_plus':
                self.scale_rewards = 'batch'
            else:
                raise ValueError(f'Invalid advantage_estimator: {self.advantage_estimator}')

    def _init_rollout(self):
        if self.rlhf_type not in rlhf_support_vllm_types:
            return

        if self.vllm_mode is not None and not self.use_vllm:
            raise ValueError('vllm_mode is not supported when use_vllm is false')

        if self.vllm_mode is None and self.use_vllm:
            raise ValueError('vllm_mode is required when use_vllm is true')

        self._init_external_vllm()

        if self.vllm_mode == 'server':
            assert not self.use_vllm or self.vllm_server_host is not None or self.vllm_server_base_url is not None

        if self.async_generate:
            assert self.vllm_mode == 'server', 'async generate require vllm_mode == server, '
            'please deploy vLLM server by `swift rollout` and assign with `vllm_server_host` '
            'for more infomations, please check '
            'https://swift.readthedocs.io/en/latest/Instruction/GRPO/getstarted/GRPO.html'

        if not self.use_vllm and self.vllm_tensor_parallel_size != 1:
            self.vllm_tensor_parallel_size = 1
            logger.warning('set vllm_tensor_parallel_size to 1 since use_vllm false')
        self._external_vllm_warning()

    def _init_padding_side(self):
        if self.rlhf_type in {'ppo', 'gkd'}:
            self.padding_side = 'left'
            # TODO: streaming, MLLM

    def _init_max_completion_length(self):
        max_completion_length = self.response_length or self.max_new_tokens or self.max_completion_length
        self.max_completion_length = self.max_new_tokens = self.response_length = max_completion_length

    def _init_metric_for_best_model(self):
        if self.rlhf_type not in {'ppo', 'grpo'}:
            super()._init_metric_for_best_model()
        elif self.rlhf_type == 'grpo' and self.metric_for_best_model is None:
            self.metric_for_best_model = 'reward'

    def _init_simpo(self):
        if self.rlhf_type != 'simpo':
            return

        self.rlhf_type = 'cpo'
        if self.loss_type is None:
            self.loss_type = 'simpo'
        if self.beta is None:
            self.beta = 2.

    def _init_rm(self):
        if self.rlhf_type == 'rm':
            self.task_type = 'seq_cls'
            self.num_labels = 1

    def _init_external_vllm(self):
        if self.rlhf_type not in rlhf_support_vllm_types or (self.vllm_server_host is None
                                                             and self.vllm_server_base_url is None):
            return
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        if is_master():
            logger.info('Start connecting to vLLM server')
            self.vllm_client = VLLMClient(
                base_urls=self.vllm_server_base_url,
                hosts=self.vllm_server_host,
                server_ports=self.vllm_server_port,
                group_ports=self.vllm_server_group_port,
                connection_timeout=self.vllm_server_timeout)
            self.vllm_client.close_communicator()
            self.vllm_client.init_communicator(device=get_current_device())
            logger.info('Connected to vLLM server')

    def _set_default(self):
        if self.beta is None:
            if self.rlhf_type == 'gkd':
                self.beta = 0.5
            else:
                self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None
            elif self.rlhf_type in ['kto']:
                self.loss_type = 'kto'
            elif self.rlhf_type == 'grpo':
                self.loss_type = 'grpo'
        if self.gradient_accumulation_steps is None:
            if self.rlhf_type == 'grpo':
                self.gradient_accumulation_steps = 1
                logger.info('Setting default gradient_accumulation_steps to 1 for GRPO.')

    def _check_grpo(self):
        if self.rlhf_type != 'grpo':
            return
        from packaging import version
        import importlib.metadata

        import trl
        trl_version = version.parse(trl.__version__)
        assert trl_version >= version.parse('0.17'), ('Your current version of `trl` is outdated. '
                                                      'Please update it by running: pip install -U trl')
        if is_mp() and self.use_vllm:
            raise ValueError('GRPO with vLLM is not compatible with `device_map`. '
                             'Please set NPROC_PER_NODE equal to num_processes.')
        if self.use_liger_kernel:
            liger_kernel_version = version.parse(importlib.metadata.version('liger-kernel'))
            assert trl_version >= version.parse('0.18')
            if self.delta is not None:
                raise ValueError('Liger loss does not support two-sided GRPO loss yet.')
            if self.sequence_parallel_size > 1:
                raise ValueError('Liger loss does not support sequence parallel yet.')
            if self.padding_free:
                raise ValueError('Liger loss does not support padding free yet.')
            if self.top_entropy_quantile < 1.0:
                raise ValueError('Liger loss does not support entropy mask yet.')
            if self.log_entropy:
                raise ValueError('Liger loss does not support log entropy yet.')
            if self.importance_sampling_level != 'token':
                if liger_kernel_version < version.parse('0.6.3'):
                    raise ValueError('Please update liger-kernel to 0.6.3 or later')
                if self.importance_sampling_level == 'sequence_token':
                    self.importance_sampling_level = 'sequence'
                    logger.info('Remapping `importance_sampling_level` from `sequence_token` to `sequence` for '
                                'liger-kernel compatibility. The two methods are computationally equivalent.')
            if self.advantage_estimator != 'grpo':
                raise ValueError('Liger loss currently only support grpo advantage estimator')
            from trl.import_utils import is_liger_kernel_available
            assert is_liger_kernel_available(), (
                'Please install/update liger-kernel by running: pip install -U liger-kernel')

        if self.async_generate and self.multi_turn_scheduler is not None:
            raise NotImplementedError('Currently, async_generate is not supported with multi-turn functionality.')

        if self.generation_batch_size or self.steps_per_generation:
            from trl.trainer.grpo_config import GRPOConfig
            assert 'generation_batch_size' in GRPOConfig.__dict__, (
                'generation_batch_size or steps_per_generation needs trl >= 0.18, '
                'please install trl `pip install trl>=0.18')

    def _external_vllm_warning(self):
        if self.rlhf_type not in rlhf_support_vllm_types or not self.vllm_server_host:
            return

        if self.vllm_max_model_len is not None:
            logger.warning(
                "Configuration conflict: 'vllm_max_model_len=%s' is ignored for external vLLM. "
                'Please specify it when launching the inference service: '
                '`swift rollout --vllm_max_model_len <value>`', self.vllm_max_model_len)

    def _check_padding_free(self):
        super()._check_padding_free()
        if self.padding_free or self.packing:
            supported_types = ['grpo', 'dpo', 'kto', 'gkd']
            if self.rlhf_type not in supported_types:
                raise NotImplementedError(
                    f"The current rlhf_type '{self.rlhf_type}' does not support padding_free/packing. "
                    'Please set --padding_free/packing to false.')

    def _check_sequence_parallel(self):
        if self.sequence_parallel_size > 1:
            supported_types = ['grpo', 'dpo']
            if self.rlhf_type not in supported_types:
                raise NotImplementedError(
                    f"The current rlhf_type '{self.rlhf_type}' does not support sequence_parallel. "
                    'Please set --sequence_parallel_size to 1.')

    def _init_teacher_deepspeed(self):
        """Initialize teacher_deepspeed configuration similar to _init_deepspeed in TrainArguments"""
        if not self.teacher_deepspeed:
            return

        # Get the same ds_config_folder as main model
        ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))
        deepspeed_mapping = {
            name: f'{name}.json'
            for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
        }

        # Check if teacher_deepspeed is a predefined name
        for ds_name, ds_config in deepspeed_mapping.items():
            if self.teacher_deepspeed == ds_name:
                self.teacher_deepspeed = os.path.join(ds_config_folder, ds_config)
                break

        # Parse the config file to dict
        self.teacher_deepspeed = json_parse_to_dict(self.teacher_deepspeed)
        logger.info(f'Using teacher_deepspeed config: {self.teacher_deepspeed}')

    def _check_gkd(self):
        if self.rlhf_type != 'gkd':
            return
        if is_mp() and self.use_vllm:
            raise ValueError('GKD with vLLM is not compatible with `device_map`. '
                             'Please set NPROC_PER_NODE equal to num_processes.')

        if self.multi_turn_scheduler is not None:
            raise NotImplementedError('Currently, multi_turn_scheduler is not supported for GKD.')

        if self.async_generate:
            raise NotImplementedError('Currently, async_generate is not supported for GKD.')
