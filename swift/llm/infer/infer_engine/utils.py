# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from queue import Queue
from typing import List

import torch
from transformers import GenerationConfig, LogitsProcessor
from transformers.generation.streamers import BaseStreamer

from swift.llm.model.register import fix_do_sample_warning
from ..protocol import RequestConfig


@dataclass
class AdapterRequest:
    name: str
    path: str


class InferTools:

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # copy from transformers.generation.streamers.TextStreamer
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False


class InferStreamer(InferTools):

    def __init__(self, template, **decode_kwargs):
        self.template = template
        self.tokenizer = template.tokenizer

        self.token_cache = []  # Reduce the time of tokenizer.decode
        self.cache_idx = 0
        self.print_idx = 0
        self.decode_kwargs = decode_kwargs
        self.first_num_space = -1  # The number of whitespace characters before the first token.

    def _align_blank_suffix(self, response: str) -> str:
        # Avoid the occurrence of repeated words in sentence.
        cur_num_space = len(response) - len(response.lstrip(' '))
        if self.first_num_space == -1:
            self.first_num_space = cur_num_space
        elif cur_num_space < self.first_num_space:
            response = ' ' * (self.first_num_space - cur_num_space) + response
        elif cur_num_space > self.first_num_space:
            response = response[cur_num_space - self.first_num_space:]
        return response

    def _get_response(self, response: str, is_finished: bool) -> str:
        # After the symbol for a new line, we flush the cache.
        if response.endswith('\n') or is_finished:
            printable_text = response[self.print_idx:]
            self.cache_idx += len(self.token_cache)
            self.print_idx = 0
        # If the last token is a CJK character, we print the characters.
        elif len(response) > 0 and self._is_chinese_char(ord(response[-1])):
            printable_text = response[self.print_idx:]
            self.print_idx += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = response[self.print_idx:response.rfind(' ') + 1]
            self.print_idx += len(printable_text)
        return printable_text

    def get_printable_text(self, raw_tokens: List[int], is_finished: bool) -> str:
        self.token_cache = raw_tokens[self.cache_idx:]
        response = self.template.decode(self.token_cache, is_finished, tokenizer_kwargs=self.decode_kwargs)
        response = self._align_blank_suffix(response)
        return self._get_response(response, is_finished)


class StreamerMixin:

    def __init__(self):
        self.queue = Queue()

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        value = self.queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


class TokensIteratorStreamer(StreamerMixin, BaseStreamer):

    def put(self, value: torch.Tensor) -> None:
        self.queue.put(value)

    def end(self) -> None:
        self.queue.put(None)


class LogitsStreamer(LogitsProcessor):

    def __init__(self):
        self.queue = Queue()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.queue.put(scores)
        return scores


def _set_generation_config_default_value(model_generation_config: GenerationConfig,
                                         generation_config: GenerationConfig) -> GenerationConfig:
    for k, v in model_generation_config.to_dict().items():
        new_v = getattr(generation_config, k, None)
        if k in ['max_length']:
            continue
        if k in ['no_repeat_ngram_size'] or v is not None and new_v is None:
            setattr(generation_config, k, v)
    return generation_config


def prepare_generation_config(model_generation_config: GenerationConfig, request_config: RequestConfig,
                              tokenizer) -> GenerationConfig:
    kwargs = {'max_new_tokens': request_config.max_tokens}
    # not use: 'n', 'best_of', 'frequency_penalty', 'presence_penalty'
    for key in ['length_penalty']:
        kwargs[key] = getattr(request_config, key)
    for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams']:
        new_value = getattr(request_config, key)
        if new_value is None:
            kwargs[key] = getattr(model_generation_config, key)
        else:
            kwargs[key] = new_value

    if not model_generation_config.do_sample:
        kwargs['temperature'] = 0
    if kwargs['temperature'] == 0:
        kwargs['do_sample'] = False
        kwargs['temperature'] = 1
        kwargs['top_p'] = 1
        kwargs['top_k'] = 50
    else:
        kwargs['do_sample'] = True
    generation_config = GenerationConfig(**kwargs)
    generation_config = _set_generation_config_default_value(model_generation_config, generation_config)
    fix_do_sample_warning(generation_config)

    if generation_config.eos_token_id is None:
        generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id
    return generation_config
