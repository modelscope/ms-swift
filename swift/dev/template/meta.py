# Copyright (c) ModelScope Contributors. All rights reserved.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Type

from transformers import PreTrainedTokenizerBase

from .base import Template
from .utils import Prompt, Word


@dataclass
class TemplateMeta:
    """The five rendering pieces a family declares, shown as chatml's concatenation:

    <s>                                     # prefix (or system_prefix: <s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n)
    <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n   # prompt
    {{RESPONSE}}<|im_end|>\n                # chat_sep between turns...
    <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n   # ...prompt again...
    {{RESPONSE}}<|im_end|>                  # ...suffix ends the final response
    """

    template_type: str
    prefix: Prompt
    prompt: Prompt
    chat_sep: Optional[Prompt]
    suffix: Prompt = field(default_factory=lambda: [["eos_token_id"]])
    template_cls: Type[Template] = Template
    system_prefix: Optional[Prompt] = None
    default_system: Optional[str] = None

    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=list)
    agent_template: Optional[str] = None
    # thinking
    is_thinking: bool = False  # the template thinks by default: enable_thinking defaults on
    thinking_prefix: str = ""  # response prefix when thinking is enabled
    non_thinking_prefix: str = ""  # response prefix without thinking (the empty thinking block)
    history_thinking_prefix: str = ""  # head of a history turn once its thinking is removed

    def to_generate_template_meta(self) -> "TemplateMeta":
        """The generation-only variant for ``use_chat_template=False``: bare ``{{QUERY}}`` prompt, bos on."""
        self = deepcopy(self)  # stop_words: the copy keeps init()'s eos append off the registry meta
        return TemplateMeta(
            self.template_type,
            prefix=[],
            prompt=["{{QUERY}}"],
            chat_sep=None,
            template_cls=self.template_cls,
            auto_add_bos=True,
            stop_words=self.stop_words,
        )

    @staticmethod
    def _has_system(prefix_or_prompt: Prompt) -> bool:
        return any("{{SYSTEM}}" in p for p in prefix_or_prompt)

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        return [p.replace("{{SYSTEM}}", "") if isinstance(p, str) else p for p in prefix]

    def _check_template_meta(self):
        # check
        for x in [self.prefix, self.prompt, self.suffix]:
            assert isinstance(x, list)
        for x in [self.chat_sep, self.system_prefix]:
            assert x is None or isinstance(x, list)

    def __post_init__(self):
        if self._has_system(self.prefix):
            assert self.system_prefix is None, "The prefix already contains {{SYSTEM}}."
            self.system_prefix = self.prefix
            self.prefix = self._replace_system(self.prefix)

        self.is_post_system = self._has_system(self.prompt)  # mistral_nemo
        if self.is_post_system:
            self.system_prompt = self.prompt
            self.prompt = [context for context in self.prompt if "{{SYSTEM}}" not in context]

        if self.system_prefix is None and not self.is_post_system:
            self.support_system = False
        else:
            self.support_system = True
        self.check_system(self.default_system)

        self.support_multi_round = self.chat_sep is not None

    @staticmethod
    def _token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Resolve token-attribute names to ids, e.g. [['eos_token_id']] -> [[2]]."""
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                v = [getattr(tokenizer, sub_v) if isinstance(sub_v, str) else sub_v for sub_v in v]
            res_value.append(v)
        return res_value

    def init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        for key in ["prefix", "prompt", "chat_sep", "suffix", "system_prefix"]:
            value = getattr(self, key)
            value = self._token_attr_to_id(tokenizer, value)
            setattr(self, key, value)

        suffix_stop = self.suffix[-1] if self.suffix else None
        if isinstance(suffix_stop, str):
            suffix_stop = suffix_stop.strip()
        self.suffix_stop = suffix_stop
        if suffix_stop and suffix_stop not in self.stop_words:
            self.stop_words.append(suffix_stop)
        if tokenizer.eos_token not in self.stop_words:
            self.stop_words.append(tokenizer.eos_token)

        self.stop_token_id = tokenizer.eos_token_id
        if suffix_stop:
            if isinstance(suffix_stop, str):
                stop_token_id = tokenizer.convert_tokens_to_ids(suffix_stop)
            elif isinstance(suffix_stop, list) and len(suffix_stop) == 1:
                stop_token_id = suffix_stop[0]
            else:
                stop_token_id = None
            if stop_token_id is not None:
                self.stop_token_id = stop_token_id

    def check_system(self, system: Optional[str]) -> None:
        if system is not None and not self.support_system:
            raise NotImplementedError(
                f"The template does not support `system`, template_type: {self.template_type}, system: {system}"
            )
