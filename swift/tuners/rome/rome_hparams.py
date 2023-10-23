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
            fact_token="subject_last",
            v_num_grad_steps=20,
            v_lr=1e-1,
            v_weight_decay=1e-2,
            clamp_norm_factor=4,
            kl_factor=0.0625,
        )
        if name == "llama-7b":
            r"""
            Supports: LLaMA-7B, LLaMA-2-7B, Baichuan-7B, InternLM-7B...
            """
            data.update(dict(
                layers=[5],
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                mlp_module_tmp="model.layers.{}.mlp",
            ))
        elif name == "llama-13b":
            r"""
            Supports LLaMA-13B, LLaMA-2-13B, Baichuan-13B...
            """
            data.update(dict(
                layers=[10],
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                mlp_module_tmp="model.layers.{}.mlp",
            ))
        else:
            raise NotImplementedError

        return cls(**data)
