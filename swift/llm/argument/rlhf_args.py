# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.llm import MODEL_MAPPING
from swift.trainers.arguments import GRPOArgumentsMixin, RLHFArgumentsMixin
from swift.utils import get_logger, is_master, set_default_ddp_config
from .train_args import TrainArguments

logger = get_logger()


@dataclass
class RewardModelArguments:
    reward_model: Optional[List[str]] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[List[str]] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[List[str]] = None


@dataclass
class TeacherModelArguments:
    teacher_model: Optional[str] = None
    teacher_adapters: List[str] = field(default_factory=list)
    teacher_model_type: Optional[List[str]] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    teacher_model_revision: Optional[List[str]] = None


@dataclass
class PPOArguments:
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
    num_generations: int = 8  # G in the GRPO paper
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    log_completions: bool = False

    # vLLM in GRPO
    use_vllm: bool = False

    # multi step
    num_iterations: int = 1

    truncation_strategy: Literal['delete', 'left', 'right', None] = None


@dataclass
class RLHFArguments(TeacherModelArguments, GRPOArguments, PPOArguments, RewardModelArguments, RLHFArgumentsMixin,
                    TrainArguments):
    """
    RLHFArguments is a dataclass that holds arguments specific to the Reinforcement
        Learning with Human Feedback (RLHF) training backend.

    Args:
        rlhf_type (Literal): Specifies the type of RLHF to use. Default is 'dpo'.
            Allowed values are 'dpo', 'orpo', 'simpo', 'kto', 'cpo'.
        ref_model_type (Optional[str]): Type of reference model. Default is None.
        ref_model_revision (Optional[str]): Revision of the reference model. Default is None.
        beta (Optional[float]): Beta parameter for RLHF. Default is None.
        label_smoothing (float): Label smoothing value. Default is 0.
        rpo_alpha (float): Alpha parameter for RPO. Default is 1.
        cpo_alpha (float): Alpha parameter for CPO. Default is 1.
        simpo_gamma (float): Gamma parameter for SimPO. Default is 1.
        desirable_weight (float): Weight for desirable outcomes in KTO. Default is 1.0.
        undesirable_weight (float): Weight for undesirable outcomes in KTO. Default is 1.0.
    """
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'] = 'dpo'
    ref_model: Optional[str] = None
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
    max_completion_length: int = 512
    loss_scale: Optional[str] = None  # 'last_round'
    # DPO
    rpo_alpha: float = 1.
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
    # compat
    max_new_tokens: Optional[int] = None  # use max_completion_length instead

    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:
        if self.rlhf_type == 'ppo':
            training_args['world_size'] = self.global_world_size

    def __post_init__(self):
        self._deprecated_warning()
        self._init_grpo()
        self._init_rm()
        self._init_simpo()
        self._init_max_completion_length()
        self._init_padding_side()
        self._set_default()
        self._init_external_vllm()
        super().__post_init__()
        self._check_grpo()
        self._external_vllm_warning()

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
        if self.rlhf_type == 'grpo' and self.beta == 0.0:
            self.ref_model = None
        elif self.rlhf_type in ['dpo', 'kto', 'ppo', 'grpo'] and self.train_type == 'full':
            self.ref_model = self.ref_model or self.model
            self.ref_model_type = self.ref_model_type or self.model_type
            self.ref_model_revision = self.ref_model_revision or self.model_revision
        elif self.ref_model is not None:
            raise ValueError('CPO/ORPO or LoRA training does not require a ref_model to be passed in.')

    def _init_grpo(self):
        if self.rlhf_type == 'grpo':
            if self.use_vllm:
                os.environ['USE_FAST_INFERENCE'] = '1'
                set_default_ddp_config()
            if self.async_generate or not self.use_vllm:
                self.sleep_level = 0
            self.remove_unused_columns = False
            logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
            if self.truncation_strategy is None:
                self.truncation_strategy = 'left'
            assert self.truncation_strategy == 'left', \
                "GRPO requires `truncation_strategy='left'`," \
                f"Current value: `truncation_strategy='{self.truncation_strategy}'`."
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
            if self.use_vllm:
                # set vllm mode
                if self.vllm_server_host is not None:
                    if self.vllm_mode != 'server':
                        self.vllm_mode = 'server'
                        logger.warning('set vllm_mode to `server` since vllm_server_host is provided')
                else:
                    if self.vllm_mode != 'colocate':
                        self.vllm_mode = 'colocate'
                        logger.warning('set vllm_mode to `colocate` since vllm_server_host is not provided')

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
        if self.rlhf_type != 'grpo' or self.vllm_server_host is None:
            return
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
        if is_master():
            self.vllm_client = VLLMClient(
                base_url=self.vllm_server_base_url,
                host=self.vllm_server_host,
                server_port=self.vllm_server_port,
                connection_timeout=self.vllm_server_timeout)
            self.vllm_client.init_communicator()

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

    def _check_grpo(self):
        if self.rlhf_type != 'grpo':
            return
        from packaging import version

        import trl
        trl_version = version.parse(trl.__version__)
        assert trl_version >= version.parse('0.17'), ('Your current version of `trl` is outdated. '
                                                      'Please update it by running: pip install -U trl')

        if self.use_liger_kernel:
            assert trl_version >= version.parse('0.18')
            if self.delta is not None:
                raise ValueError('Liger loss does not support two-sided GRPO loss yet.')
            from trl.import_utils import is_liger_kernel_available
            assert is_liger_kernel_available(), (
                'Please install/update liger-kernel by running: pip install -U liger-kernel')

        if self.vllm_mode == 'server':
            assert not self.use_vllm or self.vllm_server_host is not None

        if self.async_generate:
            assert self.vllm_mode == 'server', 'async generate require vllm_mode == server, '
            'please deploy vLLM server by `swift rollout` and assign with `vllm_server_host` '
            'for more infomations, please check '
            'https://swift.readthedocs.io/en/latest/Instruction/GRPO/getstarted/GRPO.html'

        if not self.use_vllm and self.vllm_tensor_parallel_size != 1:
            self.vllm_tensor_parallel_size = 1
            logger.warning('set vllm_tensor_parallel_size to 1 since use_vllm false')

        if self.async_generate and self.multi_turn_scheduler is not None:
            raise NotImplementedError('Currently, async_generate is not supported with multi-turn functionality.')

        if self.generation_batch_size or self.steps_per_generation:
            from trl.trainer.grpo_config import GRPOConfig
            assert 'generation_batch_size' in GRPOConfig.__dict__, (
                'generation_batch_size or steps_per_generation needs trl >= 0.18, '
                'please install trl `pip install trl>=0.18')

    def _external_vllm_warning(self):
        if self.rlhf_type != 'grpo' or not self.vllm_server_host:
            return

        if self.vllm_device is not None:
            logger.warning("Configuration conflict: External vLLM engine detected, but 'vllm_device' is set to '%s'. ",
                           self.vllm_device)

        if self.vllm_max_model_len is not None:
            logger.warning(
                "Configuration conflict: 'vllm_max_model_len=%s' is ignored for external vLLM. "
                'Please specify it when launching the inference service: '
                '`swift rollout --max_model_len <value>`', self.vllm_max_model_len)

    def _deprecated_warning(self):
        if self.rlhf_type != 'grpo':
            return

        if self.tensor_parallel_size is not None:
            logger.warning(
                "The parameter 'tensor_parallel_size' has been deprecated and will be removed in version 3.6. "
                "It is recommended to use 'vllm_tensor_parallel_size' instead.")
            self.vllm_tensor_parallel_size = self.tensor_parallel_size

        if self.vllm_device is not None:
            logger.warning("The parameter 'vllm_device' has been deprecated and will be removed in version 3.6. ")

        if self.vllm_max_num_seqs is not None:
            logger.warning("The parameter 'vllm_max_num_seqs' is automatically set, "
                           'and has been deprecated and will be removed in version 3.6. ')

        if self.num_infer_workers is not None:
            logger.warning(
                "The parameter 'num_infer_workers' has been deprecated and will be removed in version 3.6. "
                'If you wish to use colocate mode, please use `vllm_mode colocate` instead. '
                'If you wish to use async mode, please use `vllm_mode server` and external vLLM server instead.')

        if self.multi_turn_func:
            logger.warning("The parameter 'multi_turn_func' has been deprecated and will be removed in version 3.7. "
                           "Please use 'multi_turn_scheduler' instead")

            self.multi_turn_scheduler = self.multi_turn_func
