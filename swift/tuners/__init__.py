# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .adapter import Adapter, AdapterConfig, AdapterModule
    from .base import SwiftModel, Swift
    from .lora import LoRA, LoRAConfig
    from .mapping import SWIFT_MAPPING, SwiftTuners
    from .side import Side, SideConfig, SideModule
    from .neftune import NEFTune, NEFTuneConfig
    from .longlora.longlora import LongLoRAModelType, LongLoRAConfig, LongLoRA
    from .restuning import ResTuning, ResTuningConfig, ResTuningBypassModule
    from .reft import Reft, ReftConfig
    from .llamapro import LLaMAPro, LLaMAProConfig
    from .peft import (AdaLoraConfig, LoftQConfig, LoHaConfig, LoKrConfig, LoraConfig, VeraConfig, BOFTConfig,
                       OFTConfig, PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                       PeftModelForSequenceClassification, PeftModelForTokenClassification, PrefixTuningConfig,
                       PromptEncoderConfig, PromptLearningConfig, PromptTuningConfig, get_peft_config, get_peft_model,
                       get_peft_model_state_dict)
    from .prompt import Prompt, PromptConfig, PromptModule
    from .scetuning.scetuning import SCETuning, SCETuningConfig
    from .utils import SwiftConfig, SwiftOutput, swift_to_peft_format
else:
    _import_structure = {
        'adapter': ['Adapter', 'AdapterConfig', 'AdapterModule'],
        'base': ['SwiftModel', 'Swift'],
        'lora': ['LoRA', 'LoRAConfig'],
        'longlora.longlora': ['LongLoRAModelType', 'LongLoRAConfig', 'LongLoRA'],
        'mapping': ['SWIFT_MAPPING', 'SwiftTuners'],
        'side': ['Side', 'SideConfig', 'SideModule'],
        'reft': ['Reft', 'ReftConfig'],
        'llamapro': ['LLaMAPro', 'LLaMAProConfig'],
        'neftune': ['NEFTune', 'NEFTuneConfig'],
        'restuning': ['ResTuning', 'ResTuningConfig', 'ResTuningBypassModule'],
        'peft': [
            'AdaLoraConfig', 'LoftQConfig', 'LoHaConfig', 'LoKrConfig', 'LoraConfig', 'VeraConfig', 'BOFTConfig',
            'OFTConfig', 'PeftConfig', 'PeftModel', 'PeftModelForCausalLM', 'PeftModelForSeq2SeqLM',
            'PeftModelForSequenceClassification', 'PeftModelForTokenClassification', 'PrefixTuningConfig',
            'PromptEncoderConfig', 'PromptLearningConfig', 'PromptTuningConfig', 'get_peft_config', 'get_peft_model',
            'get_peft_model_state_dict'
        ],
        'prompt': ['Prompt', 'PromptConfig', 'PromptModule'],
        'scetuning': ['SCETuning', 'SCETuningConfig'],
        'utils': ['SwiftConfig', 'SwiftOutput', 'swift_to_peft_format'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
