# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from kmeng01/rome.
from dataclasses import dataclass
from typing import List

from .hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    # Method
    layers: List[int]
    fact_token: str
    v_num_grad_steps: int
    v_lr: float
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float

    # Module templates
    rewrite_module_tmp: str
    mlp_module_tmp: str

    @classmethod
    def from_name(cls, name: str):
        data = dict(
            fact_token='subject_last',
            v_num_grad_steps=20,
            v_lr=1e-1,
            v_weight_decay=1e-2,
            clamp_norm_factor=4,
            kl_factor=0.0625,
        )
        if name == 'llama-7b':
            data.update(
                dict(
                    layers=[5],
                    rewrite_module_tmp='model.layers.{}.mlp.down_proj',
                    mlp_module_tmp='model.layers.{}.mlp',
                ))
        elif name == 'llama-13b':
            data.update(
                dict(
                    layers=[10],
                    rewrite_module_tmp='model.layers.{}.mlp.down_proj',
                    mlp_module_tmp='model.layers.{}.mlp',
                ))
        elif name == 'chatglm-6b':
            data.update(
                dict(
                    layers=[5],
                    rewrite_module_tmp='transformer.encoder.layers.{}.mlp.dense_4h_to_h',
                    mlp_module_tmp='transformer.encoder.layers.{}.mlp',
                ))
        else:
            raise NotImplementedError(f'{name} not supported.')

        return cls(**data)
