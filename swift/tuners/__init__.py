# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .adapter import Adapter, AdapterConfig, AdapterModule
    from .base import SwiftModel, Swift
    from .lora import LoRA, LoRAConfig
    from .mapping import SWIFT_MAPPING
    from .side import Side, SideConfig, SideModule
    from .restuning import ResTuning, ResTuningConfig, ResTuningModule
    from .peft import (LoraConfig, PeftConfig, PeftModel, PeftModelForCausalLM,
                       PeftModelForSeq2SeqLM,
                       PeftModelForSequenceClassification,
                       PeftModelForTokenClassification, PrefixTuningConfig,
                       PromptEncoderConfig, PromptLearningConfig,
                       PromptTuningConfig, get_peft_config, get_peft_model,
                       get_peft_model_state_dict)
    from .prompt import Prompt, PromptConfig, PromptModule
    from .utils import SwiftConfig, SwiftOutput
else:
    _import_structure = {
        'adapter': ['Adapter', 'AdapterConfig', 'AdapterModule'],
        'base': ['SwiftModel', 'Swift'],
        'lora': ['LoRA', 'LoRAConfig'],
        'mapping': ['SWIFT_MAPPING'],
        'side': ['Side', 'SideConfig', 'SideModule'],
        'restuning': ['ResTuning', 'ResTuningConfig', 'ResTuningModule'],
        'peft': [
            'LoraConfig', 'PeftConfig', 'PeftModel', 'PeftModelForCausalLM',
            'PeftModelForSeq2SeqLM', 'PeftModelForSequenceClassification',
            'PeftModelForTokenClassification', 'PrefixTuningConfig',
            'PromptEncoderConfig', 'PromptLearningConfig',
            'PromptTuningConfig', 'get_peft_config', 'get_peft_model',
            'get_peft_model_state_dict'
        ],
        'prompt': ['Prompt', 'PromptConfig', 'PromptModule'],
        'utils': ['SwiftConfig', 'SwiftOutput'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
