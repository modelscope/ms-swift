# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Literal, Optional

from swift.llm import MODEL_MAPPING
from .train_args import TrainArguments


@dataclass
class RLHFArguments(TrainArguments):
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
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm'] = 'dpo'
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
        self._init_simpo()
        self._set_default()
        super().__post_init__()

        if self.rlhf_type not in ['cpo', 'orpo', 'rm'] and (self.train_type == 'full' or self.rlhf_type == 'ppo'):
            self.ref_model = self.ref_model or self.model
            self.ref_model_type = self.ref_model_type or self.model_type
            self.ref_model_revision = self.ref_model_revision or self.model_revision
        elif self.ref_model is not None:
            raise ValueError('CPO/ORPO or LoRA training does not require a ref_model to be passed in.')

    def _init_simpo(self):
        if self.rlhf_type != 'simpo':
            return

        self.rlhf_type = 'cpo'
        if self.loss_type is None:
            self.loss_type = 'simpo'
        if self.beta is None:
            self.beta = 2.

    def _set_default(self):
        if self.beta is None:
            self.beta = 0.1
        if self.loss_type is None:
            if self.rlhf_type in ['dpo', 'cpo']:
                self.loss_type = 'sigmoid'  # else None
            elif self.rlhf_type in ['kto']:
                self.loss_type = 'kto'
