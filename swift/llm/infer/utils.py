# Copyright (c) Alibaba, Inc. and its affiliates.

import time
from queue import Queue
from typing import List

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria, LogitsProcessor
from transformers.generation.streamers import BaseStreamer

from swift.plugin import Metric
from ..template import Template, Word


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

    @staticmethod
    def _skip_stop_tokens(generate_ids: List[int], stop_tokens: List[int], is_finished: bool) -> List[int]:
        len_tokens = len(stop_tokens)
        if is_finished and generate_ids[-len_tokens:] == stop_tokens:
            return generate_ids[:-len_tokens]
        if not is_finished:
            for i in range(len_tokens, 0, -1):
                if generate_ids[-i:] == stop_tokens[:i]:
                    return generate_ids[:-i]
        return generate_ids

    @staticmethod
    def safe_decode(template: Template, generate_ids: List[int], is_finished: bool, **decode_kwargs) -> str:
        # Do not print template.suffix[-1] and eos_token.
        tokenizer = template.tokenizer

        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:-1]
        # skip suffix and eos_token
        template_suffix = template.suffix[-1]
        if isinstance(template_suffix, str):
            template_suffix = tokenizer.encode(template_suffix, add_special_tokens=False)
        generate_ids = InferTools._skip_stop_tokens(generate_ids, template_suffix, is_finished)
        return tokenizer.decode(generate_ids, **decode_kwargs)
        # if not is_finished or is_finished and response[-len_suffix:] == template_suffix:
        #     # To avoid response length being shorter than previous response length during streaming.
        #     # TODO:check
        #     # idx = max(len(response) - len_suffix, 0, self.print_idx)
        #     response = response[:-len_suffix]


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
        response = self.safe_decode(self.template, self.token_cache, is_finished, **self.decode_kwargs)
        response = self._align_blank_suffix(response)
        return self._get_response(response, is_finished)


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class TokensIteratorStreamer(BaseStreamer):

    def __init__(self):
        self.queue = Queue()  # Queue[int]

    def put(self, value: torch.Tensor) -> None:
        self.queue.put(value)

    def end(self) -> None:
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> List[int]:
        value = self.queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value


class StopWordsCriteria(StoppingCriteria):
    # The returned sentence includes stop words.
    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: List[Word], **decode_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.decode_kwargs = decode_kwargs
        self.start_idx = -1

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.start_idx == -1:
            self.start_idx = input_ids.shape[1] - 1

        is_finished = torch.zeros((input_ids.shape[0], ), device=input_ids.device, dtype=torch.bool)
        for i in range(input_ids.shape[0]):
            # [-20:]: Assuming the end tokens do not exceed 20 tokens,
            #   to avoid input_ids being too long and affecting efficiency.
            text = self.tokenizer.decode(input_ids[i, self.start_idx:][-20:], **self.decode_kwargs)
            for stop_word in self.stop_words:
                if isinstance(stop_word, str) and stop_word in text or isinstance(
                        stop_word, list) and input_ids[i][-len(stop_word):].tolist() == stop_word:
                    is_finished[i] = True
                    break
            else:
                is_finished[i] = False
        return is_finished
