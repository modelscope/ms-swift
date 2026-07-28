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

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        logger.info('[IMPORTANT] Nemotron-H is a hybrid Mamba2 + Attention + MoE model. For best speed, install '
                    '`causal-conv1d>=1.2.0` and `mamba-ssm`; otherwise it falls back to a slower naive kernel.')
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.nemotron_h,
        [
            ModelGroup([
                Model('nv-community/EA-NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-07202026',
                      'nv-community/EA-NVIDIA-Nemotron-3.5-Nano-30B-A3B-BF16-07202026'),
                Model('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'),
            ]),
        ],
        NemotronHLoader,
        template=TemplateType.nemotron_h,
        architectures=['NemotronHForCausalLM'],
        model_arch=None,
        requires=['transformers>=5.0', 'mamba-ssm', 'causal-conv1d>=1.2.0'],
    ))
