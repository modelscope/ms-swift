# Copyright (c) ModelScope Contributors. All rights reserved.
from .dummy import DummyTuner
from .entropack import prepare_entropack_model
from .ia3 import IA3Tuner
from .lora_llm import LoRALLMTuner

tuners_map = {
    'ia3': IA3Tuner,
    'lora_llm': LoRALLMTuner,
    'dummy': DummyTuner,
}
model_prepare_hooks = {
    'entropack': prepare_entropack_model,
}


def apply_model_prepare_hooks(args, model):
    for hook in model_prepare_hooks.values():
        model = hook(args, model)
    return model
