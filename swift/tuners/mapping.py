# Copyright (c) Alibaba, Inc. and its affiliates.

from .adapter import Adapter, AdapterConfig
from .lora import LoRA, LoRAConfig
from .prompt import Prompt, PromptConfig
from .restuning import ResTuning, ResTuningConfig
from .side import Side, SideConfig


class SwiftTuners:
    ADAPTER = 'ADAPTER'
    PROMPT = 'PROMPT'
    LORA = 'LORA'
    SIDE = 'SIDE'
    RESTUNING = 'RESTUNING'


SWIFT_MAPPING = {
    SwiftTuners.ADAPTER: (AdapterConfig, Adapter),
    SwiftTuners.PROMPT: (PromptConfig, Prompt),
    SwiftTuners.LORA: (LoRAConfig, LoRA),
    SwiftTuners.SIDE: (SideConfig, Side),
    SwiftTuners.RESTUNING: (ResTuningConfig, ResTuning),
}
