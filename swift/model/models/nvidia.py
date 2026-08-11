# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedModel

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class NemotronHLoader(ModelLoader):
    default_trust_remote_code = False


register_model(
    ModelMeta(
        LLMModelType.nemotron_h,
        [
            ModelGroup([
                Model('nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
                      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'),
                Model('nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4',
                      'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4'),
            ]),
        ],
        NemotronHLoader,
        template=TemplateType.nemotron_h,
        architectures=['NemotronHForCausalLM'],
        model_arch=None,
        requires=['transformers>=5.0', 'mamba-ssm', 'causal-conv1d>=1.2.0'],
    ))
