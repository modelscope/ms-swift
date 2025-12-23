from dataclasses import dataclass, field
from typing import List, Optional

from ..constant import LLMTemplateType
from ..register import TemplateMeta, register_template
from ..utils import Prompt, Word

DEFAULT_SYSTEM = 'You are a helpful assistant'


@dataclass
class MiniMindTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<|im_end|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<|im_end|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])
    default_system: Optional[str] = DEFAULT_SYSTEM
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])


register_template(MiniMindTemplateMeta(LLMTemplateType.minimind))
