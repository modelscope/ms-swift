# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.llm import MODEL_MAPPING
from swift.trainers.arguments import GRPOArgumentsMixin
from swift.utils import get_logger
from .train_args import TrainArguments

logger = get_logger()


@dataclass
class RewardModelArguments:
    reward_model: Optional[str] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[str] = None


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
    response_length: int = 512
    missing_eos_penalty: Optional[float] = None


@dataclass
class GRPOArguments(GRPOArgumentsMixin):
    num_generations: int = 8  # G in the GRPO paper
    max_completion_length: int = 512
    ds3_gather_for_generation: bool = True
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    log_completions: bool = False

    # vLLM in GRPO
    use_vllm: bool = False
    vllm_device: List[str] = field(default_factory=lambda: ['auto'])
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: Optional[int] = None

    # multi step
    num_iterations: int = 1
    epsilon: float = 0.2


@dataclass
class RLHFArguments(GRPOArguments, PPOArguments, RewardModelArguments, TrainArguments):
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
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo'] = 'dpo'
    ref_model: Optional[str] = None
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
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
    # PPO/GRPO
    temperature: float = 0.9

    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:
        if self.rlhf_type == 'ppo':
            training_args['world_size'] = self.global_world_size

    def __post_init__(self):
        self._init_grpo()
        self._init_rm()
        self._init_simpo()
        self._init_ppo()
        self._set_default()
        super().__post_init__()
        self._init_grpo_ds3()

        if self.loss_scale is None:
            if self.rlhf_type == 'orpo' and not self.model_meta.is_multimodal:
                # Avoid padding labels during the model's forward pass in multimodal models.
                # Some multimodal models do not expand the image pad token.
                self.loss_scale = 'default'
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

    @staticmethod
    def _set_default_ddp_config():
        # It runs normally with Python as well.
        rank = int(os.getenv('RANK', -1))
        if rank == -1:
            os.environ['NPROC_PER_NODE'] = '1'
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')

    def _init_grpo(self):
        if self.rlhf_type == 'grpo':
            if self.use_vllm or self.use_lmdeploy:
                os.environ['USE_FAST_INFERENCE'] = '1'
                self._set_default_ddp_config()
            if self.async_generate or not self.use_vllm:
                self.sleep_level = 0
            if self.sleep_level > 0:
                self.gradient_accumulation_steps = 1
            self.remove_unused_columns = False
            logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
            self.truncation_strategy = 'left'  # Used for trimming the excessively long parts of a prompt.
            if self.beta is None:
                self.beta = 0.04  # https://arxiv.org/abs/2402.03300

    def _init_ppo(self):
        if self.rlhf_type == 'ppo':
            self.padding_side = 'left'
            # TODO: streaming, MLLM

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

    def _set_default(self):
        if self.beta is None:
            self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None
            elif self.rlhf_type in ['kto']:
                self.loss_type = 'kto'

    def _init_grpo_ds3(self):
        if self.rlhf_type == 'grpo' and self.deepspeed:
            if 'zero_optimization' in self.deepspeed and self.deepspeed['zero_optimization']['stage'] == 3:
                self.deepspeed['zero_optimization']['stage3_prefetch_bucket_size'] = 0
