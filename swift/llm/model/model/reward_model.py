# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import AutoConfig, AutoModel

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import RMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model

logger = get_logger()





            # ModelGroup([
            #     Model('Qwen/Qwen2.5-Math-RM-72B', 'Qwen/Qwen2.5-Math-RM-72B'),
            #     Model('Qwen/Qwen2-Math-RM-72B', 'Qwen/Qwen2-Math-RM-72B'),
            # ]),