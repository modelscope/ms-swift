# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2023-present the HuggingFace Inc. team.

import os.path
from typing import Optional

from peft import (AdaLoraConfig, IA3Config, LoftQConfig, LoHaConfig,
                  LoKrConfig, LoraConfig, OFTConfig, PeftConfig, PeftModel,
                  PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                  PeftModelForSequenceClassification,
                  PeftModelForTokenClassification, PrefixTuningConfig,
                  PromptEncoderConfig, PromptLearningConfig,
                  PromptTuningConfig, get_peft_config, get_peft_model,
                  get_peft_model_state_dict)

from swift.hub.snapshot_download import snapshot_download


def get_wrapped_class(module_class):
    """Get a custom wrapper class for peft classes to download the models from the ModelScope hub

    Args:
        module_class: The actual module class

    Returns:
        The wrapper
    """

    class PeftWrapper(module_class):

        @classmethod
        def from_pretrained(cls,
                            model,
                            model_id,
                            *args,
                            revision: Optional[str] = None,
                            **kwargs):
            if not os.path.exists(model_id):
                model_id = snapshot_download(model_id, revision=revision)
            return module_class.from_pretrained(model, model_id, *args,
                                                **kwargs)

    return PeftWrapper


def wrap_module(module):
    if not hasattr(module, 'from_pretrained'):
        return module

    return get_wrapped_class(module)


PeftModel = wrap_module(PeftModel)
PeftConfig = wrap_module(PeftConfig)
PeftModelForSeq2SeqLM = wrap_module(PeftModelForSeq2SeqLM)
PeftModelForSequenceClassification = wrap_module(
    PeftModelForSequenceClassification)
PeftModelForTokenClassification = wrap_module(PeftModelForTokenClassification)
PeftModelForCausalLM = wrap_module(PeftModelForCausalLM)
PromptEncoderConfig = wrap_module(PromptEncoderConfig)
PromptTuningConfig = wrap_module(PromptTuningConfig)
PrefixTuningConfig = wrap_module(PrefixTuningConfig)
PromptLearningConfig = wrap_module(PromptLearningConfig)
LoraConfig = wrap_module(LoraConfig)
AdaLoraConfig = wrap_module(AdaLoraConfig)
IA3Config = wrap_module(IA3Config)
LoHaConfig = wrap_module(LoHaConfig)
LoKrConfig = wrap_module(LoKrConfig)
LoftQConfig = wrap_module(LoftQConfig)
OFTConfig = wrap_module(OFTConfig)
get_peft_config = get_peft_config
get_peft_model_state_dict = get_peft_model_state_dict
get_peft_model = get_peft_model
