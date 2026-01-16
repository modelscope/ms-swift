# Copyright (c) ModelScope Contributors. All rights reserved.
from .dummy import DummyTuner
from .ia3 import IA3Tuner
from .lora_llm import LoRALLMTuner

tuners_map = {
    'ia3': IA3Tuner,
    'lora_llm': LoRALLMTuner,
    'dummy': DummyTuner,
}
