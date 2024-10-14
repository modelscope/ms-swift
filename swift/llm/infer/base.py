# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import time
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm

from swift.plugin import Metric
from ..model import get_default_torch_dtype, get_model_tokenizer
from ..template import Template
from .protocol import ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest, RequestConfig


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
    def safe_decode(template: Template, generate_ids: List[int], is_finished: bool, **decode_kwargs) -> str:
        tokenizer = template.tokenizer

        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:-1]
        response = tokenizer.decode(generate_ids, **decode_kwargs)
        if len(response) == 0 or response.endswith('\n'):
            return response
        # skip suffix and eos_token
        template_suffix = template.suffix[-1]
        if isinstance(template_suffix, list):
            template_suffix = tokenizer.decode(template_suffix)
        len_suffix = len(template.suffix[-1])
        if not is_finished or is_finished and response[-len_suffix:] == template.suffix[-1]:
            # To avoid response length being shorter than previous response length during streaming.
            # TODO:check
            # idx = max(len(response) - len_suffix, 0, self.print_idx)
            response = response[:-len_suffix]
        return response


class InferStreamer(InferTools):

    def __init__(self, template, input_delta: bool = False, **decode_kwargs):
        self.template = template
        self.tokenizer = template.tokenizer

        self.token_cache = []  # Reduce the time of tokenizer.decode
        self.input_delta = input_delta
        self.cache_idx = 0  # This parameter takes effect when input_delta is set to False.
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

    def _get_printable_text(self, response: str) -> str:
        # After the symbol for a new line, we flush the cache.
        if response.endswith('\n'):
            printable_text = response[self.print_idx:]
            if self.input_delta:
                self.token_cache = []
            else:
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

    def put(self, tokens: List[int], is_finished: bool) -> str:
        if self.input_delta:
            self.token_cache += tokens
        else:
            self.token_cache = tokens[self.cache_idx:]
        response = self.safe_decode(self.template, self.token_cache, is_finished, **self.decode_kwargs)
        response = self._align_blank_suffix(response)
        return self._get_printable_text(response)


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)
        self.add_state('num_samples', 0)

    def update(self, output):
        if output is None:
            self.num_samples += 1
            return
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': self.num_samples,
            'runtime': runtime,
            'samples/s': self.num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class InferEngine(ABC):

    def _prepare_model_tokenizer(self,
                                 model_id_or_path: str,
                                 torch_dtype: Optional[torch.dtype],
                                 load_model: bool,
                                 *,
                                 model_type: Optional[str] = None,
                                 **kwargs) -> None:
        use_hf = kwargs.pop('use_hf', None)
        revision = kwargs.pop('revision', None)
        model, tokenizer = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            revision=revision)
        config = tokenizer.config
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.torch_dtype = config.torch_dtype

        self.model_type = tokenizer.model_type
        self.model_dir = tokenizer.model_dir
        self.is_multimodal = tokenizer.is_multimodal
        self.is_moe = tokenizer.is_moe
        self.chat_template = tokenizer.chat_template
        self.generation_template = tokenizer.generation_template

    @staticmethod
    async def _run_infer(task, queue, stream: bool = False):
        if stream:
            async for stream_response in await task:
                queue.put(stream_response)
        else:
            queue.put(await task)
        queue.put(None)

    @staticmethod
    async def _batch_run(tasks):
        return await asyncio.gather(*tasks)

    @staticmethod
    def _infer_stream(tasks,
                      stream: bool = True,
                      use_tqdm: bool = True,
                      metrics: Optional[List[Metric]] = None) -> Iterator[List[ChatCompletionStreamResponse]]:
        if metrics is None:
            metrics = []
        for metric in metrics:
            metric.reset()

        queue_list = [Queue() for i in range(len(tasks))]
        new_tasks = [InferEngine._run_infer(task, queue_list[i], stream) for i, task in enumerate(tasks)]
        thread = Thread(target=lambda: asyncio.run(InferEngine._batch_run(new_tasks)))
        thread.start()

        prog_bar = tqdm(total=len(new_tasks), dynamic_ncols=True, disable=not use_tqdm)
        n_finished = 0
        outputs = [None] * len(new_tasks)

        while n_finished < len(new_tasks):
            for i, queue in enumerate(queue_list):
                output = queue.get(block=False)
                if output is None:
                    n_finished += 1
                    prog_bar.update()
                else:
                    outputs[i] = output
                for metric in metrics:
                    metric.update(output)
            yield outputs

    @staticmethod
    def _infer(tasks, use_tqdm: bool = True, metrics: Optional[List[Metric]] = None) -> List[ChatCompletionResponse]:
        for outputs in InferEngine._infer_stream(tasks, False, use_tqdm, metrics):
            pass
        return outputs

    @abstractmethod
    @torch.inference_mode()
    async def infer_async(self,
                          template: Template,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          request_id: Optional[str] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass

    @torch.inference_mode()
    def infer(self,
              template: Template,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              *,
              use_tqdm: Optional[bool] = None,
              metrics: Optional[List[Metric]] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[ChatCompletionStreamResponse]]]:
        tasks = [
            self.infer_async(template, infer_request, request_config, **kwargs) for infer_request in infer_requests
        ]
        if use_tqdm is None:
            use_tqdm = not request_config.stream
        if request_config.stream:
            return self._infer_stream(tasks, True, use_tqdm, metrics)
        else:
            return self._infer(tasks, use_tqdm, metrics)
