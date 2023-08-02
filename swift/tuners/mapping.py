# Copyright (c) Alibaba, Inc. and its affiliates.

from .adapter import Adapter, AdapterConfig
from .lora import LoRA, LoRAConfig
from .prompt import Prompt, PromptConfig


class SwiftTuners:
    ADAPTER = 'ADAPTER'
    PROMPT = 'PROMPT'
    LORA = 'LORA'


SWIFT_MAPPING = {
    SwiftTuners.ADAPTER: (AdapterConfig, Adapter),
    SwiftTuners.PROMPT: (PromptConfig, Prompt),
    SwiftTuners.LORA: (LoRAConfig, LoRA)
}
