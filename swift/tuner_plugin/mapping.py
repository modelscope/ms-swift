# Copyright (c) ModelScope Contributors. All rights reserved.
from .dummy import DummyTuner
from .ia3 import IA3Tuner
from .llm_lora import LLMLoraTuner

tuners_map = {
    'ia3': IA3Tuner,
    'llm_lora': LLMLoraTuner,
    'dummy': DummyTuner,
}
