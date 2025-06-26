# Copyright (c) Alibaba, Inc. and its affiliates.
import os
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
        padding_side: The padding_side when the training batch_size >= 2
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

    truncation_strategy: Literal['delete', 'left', 'right', None] = None
    max_pixels: Optional[int] = None
    agent_template: Optional[str] = None
    norm_bbox: Literal['norm1000', 'none', None] = None
    use_chat_template: Optional[bool] = None
    # train
    padding_free: bool = False
    padding_side: Literal['left', 'right'] = 'right'
    loss_scale: str = 'default'
    sequence_parallel_size: int = 1
    # infer/deploy
    response_prefix: Optional[str] = None
    template_backend: Literal['swift', 'jinja'] = 'swift'

    def __post_init__(self):
        if self.template is None and hasattr(self, 'model_meta'):
            self.template = self.model_meta.template
        if self.use_chat_template is None:
            self.use_chat_template = True
        if self.system is not None:
            if self.system.endswith('.txt'):
                assert os.path.isfile(self.system), f'self.system: {self.system}'
                with open(self.system, 'r') as f:
                    self.system = f.read()
            else:
                self.system = self.system.replace('\\n', '\n')
        if self.response_prefix is not None:
            self.response_prefix = self.response_prefix.replace('\\n', '\n')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'delete'

    def get_template_kwargs(self):
        truncation_strategy = self.truncation_strategy
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'
        remove_unused_columns = self.remove_unused_columns
        if hasattr(self, 'rlhf_type') and self.rlhf_type == 'grpo':
            remove_unused_columns = True
        return {
            'default_system': self.system,
            'max_length': self.max_length,
            'truncation_strategy': truncation_strategy,
            'max_pixels': self.max_pixels,
            'agent_template': self.agent_template,
            'norm_bbox': self.norm_bbox,
            'use_chat_template': self.use_chat_template,
            'remove_unused_columns': remove_unused_columns,
            # train
            'padding_free': self.padding_free,
            'padding_side': self.padding_side,
            'loss_scale': self.loss_scale,
            'sequence_parallel_size': self.sequence_parallel_size,
            # infer/deploy
            'response_prefix': self.response_prefix,
            'template_backend': self.template_backend,
        }
