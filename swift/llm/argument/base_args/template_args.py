# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TemplateArguments:
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'truncation_left'] = 'truncation_left'
    tools_prompt: str = 'react_en'  # Override the default_tools_prompt in the template.
    max_pixels: int = -1
    # train
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1

    def __post_init__(self):
        if self.template is None:
            self.template = self.model_meta.template
