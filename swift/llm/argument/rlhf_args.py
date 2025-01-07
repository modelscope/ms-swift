# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm import MODEL_MAPPING
from .train_args import TrainArguments


@dataclass
class PPOArguments:
    reward_model: Optional[str] = None
    reward_adapters: List[str] = field(default_factory=list)
    reward_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[str] = None

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
    temperature: float = 0.7
    missing_eos_penalty: Optional[float] = None


@dataclass
class RLHFArguments(PPOArguments, TrainArguments):
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
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo'] = 'dpo'
    ref_model: Optional[str] = None
    ref_model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_revision: Optional[str] = None

    beta: Optional[float] = None
    label_smoothing: float = 0
    # DPO
    rpo_alpha: float = 1.
    # CPO
    cpo_alpha: float = 1.
    # SimPO
    simpo_gamma: float = 1
    # KTO
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0

    def __post_init__(self):
        self._init_rm()
        self._init_simpo()
        self._set_default()
        super().__post_init__()
        self._init_ppo()

        if self.rlhf_type in ['dpo', 'kto'] and self.train_type == 'full' or self.rlhf_type == 'ppo':
            self.ref_model = self.ref_model or self.model
            self.ref_model_type = self.ref_model_type or self.model_type
            self.ref_model_revision = self.ref_model_revision or self.model_revision
        elif self.ref_model is not None:
            raise ValueError('CPO/ORPO or LoRA training does not require a ref_model to be passed in.')

    def _init_ppo(self):
        if self.rlhf_type == 'ppo':
            self.padding_side = 'left'
            self.metric_for_best_model = None
            self.training_args.metric_for_best_model = None
            # TODO: streaming, MLLM

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
