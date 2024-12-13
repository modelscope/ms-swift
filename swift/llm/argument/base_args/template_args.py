# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Literal, Optional

from swift.llm import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()


@dataclass
class TemplateArguments:
    """
    TemplateArguments class is a dataclass that holds various arguments related to template configuration and usage.

    Args:
        template (Optional[str]): Template type. Default is None, meaning use the template of the model_type.
        system (Optional[str]): Override the default system in the template. Default is None.
        max_length (Optional[int]): Maximum length for the template. Default is None.
        truncation_strategy (Literal): Strategy for truncating the template. Default is 'delete'.
        max_pixels (Optional[int]): Maximum number of pixels for the template. Default is None.
        tools_prompt (str): Override the default tools prompt in the template. Default is 'react_en'.
        loss_scale (str): Loss scale for training. Default is 'default',
            meaning only calculate the loss of the assistant.
        sequence_parallel_size (int): Size of sequence parallelism. Default is 1.
        use_chat_template (str): Use chat template or default generation template, default True
        template_backend (str): Use swift template or jinja
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})
    system: Optional[str] = None  # Override the default_system in the template.
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'left'] = 'delete'
    max_pixels: Optional[int] = None
    tools_prompt: str = 'react_en'  # Override the default_tools_prompt in the template.
    # train
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    # infer/deploy
    use_chat_template: bool = True
    template_backend: Literal['swift', 'jinja'] = 'swift'

    def __post_init__(self):
        if self.template is None and hasattr(self, 'model_meta'):
            self.template = self.model_meta.template

        if self.max_length is None and hasattr(self, 'model_info'):
            self.max_length = self.model_info.max_model_len

    def get_template_kwargs(self):
        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'
        return {
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'tools_prompt': self.tools_prompt,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            'template_backend': self.template_backend
        }
