# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from swift.llm import get_model_tokenizer, get_template, load_dataset
from swift.llm.template import Template, split_action_action_input
from swift.plugin import Metric
from swift.utils import get_logger
from ..protocol import (ChatCompletionMessageToolCall, ChatCompletionResponse, ChatCompletionStreamResponse, Function,
                        InferRequest, RequestConfig, UsageInfo)
from .base import BaseInferEngine

logger = get_logger()


class InferEngine(BaseInferEngine):

    def _prepare_model_tokenizer(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype],
            load_model: bool,
            *,
            model_type: Optional[str] = None,
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            # model
            model_kwargs: Optional[Dict[str, Any]] = None,
            attn_impl: Literal['flash_attn', 'sdpa', 'eager', 'auto'] = 'auto') -> None:
        model, tokenizer = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            revision=revision,
            model_kwargs=model_kwargs,
            attn_impl=attn_impl)
        self.tokenizer = tokenizer
        self.model = model
        model_info = tokenizer.model_info
        self.model_info = model_info
        self.model_dir = model_info.model_dir
        self.config = model_info.config
        self.max_model_len = model_info.max_model_len
        self.torch_dtype = model_info.torch_dtype

    def _get_stop_words(self, stop_words: List[Union[str, List[int], None]]) -> List[str]:
        stop: List[str] = []
        for stop_word in stop_words:
            if stop_word is None:
                continue
            elif isinstance(stop_word, list):
                stop_word = self.tokenizer.decode(stop_word)
            assert isinstance(stop_word, str)
            if stop_word not in stop:
                stop.append(stop_word)
        return stop

    @staticmethod
    def __infer_stream(tasks,
                       stream: bool = True,
                       use_tqdm: bool = True) -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:

        async def _run_infer(i, task, queue, stream: bool = False):
            # task with queue
            if stream:
                async for stream_response in await task:
                    queue.put((i, stream_response))
            else:
                queue.put((i, await task))
            queue.put((i, None))

        async def _batch_run(tasks):
            return await asyncio.gather(*tasks)

        queue = Queue()
        new_tasks = [_run_infer(i, task, queue, stream) for i, task in enumerate(tasks)]
        thread = Thread(target=lambda: asyncio.run(_batch_run(new_tasks)))
        thread.start()

        prog_bar = tqdm(total=len(new_tasks), dynamic_ncols=True, disable=not use_tqdm)
        n_finished = 0
        outputs = [None] * len(new_tasks)

        while n_finished < len(new_tasks):
            i, output = queue.get()
            if output is None:  # is_finished
                n_finished += 1
                prog_bar.update()
            else:
                if outputs[i] is not None:  # The logic will only apply to the stream.
                    yield outputs
                    outputs = [None] * len(new_tasks)
                outputs[i] = output
        yield outputs

    @staticmethod
    def __infer_full(tasks, use_tqdm: bool = True) -> List[ChatCompletionResponse]:
        for outputs in InferEngine.__infer_stream(tasks, False, use_tqdm):
            pass
        return outputs

    @staticmethod
    def _get_usage_info(num_prompt_tokens: int, num_generated_tokens: int) -> UsageInfo:
        return UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

    @staticmethod
    def _update_metrics(result, metrics: Optional[List[Metric]] = None):
        result_origin = result
        if not isinstance(result, (list, tuple)):
            result = [result]
        for response in result:
            if response is None:
                continue
            for metric in metrics:
                metric.update(response)
        return result_origin

    @torch.inference_mode()
    def infer(self,
              template: Template,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        tasks = [
            self.infer_async(template, infer_request, request_config, **kwargs) for infer_request in infer_requests
        ]
        if use_tqdm is None:
            use_tqdm = not request_config.stream
        if request_config.stream:

            def _gen_wrapper():
                for res in self.__infer_stream(tasks, True, use_tqdm), metrics:
                    yield InferEngine._update_metrics(res, metrics)

            return _gen_wrapper()
        else:
            return self._update_metrics(self.__infer_full(tasks, use_tqdm), metrics)

    def _get_toolcall(self, response: Union[str, List[int]],
                      is_finished: bool) -> Optional[List[ChatCompletionMessageToolCall]]:
        if not is_finished:
            return None
        if not isinstance(response, str):
            response = self.tokenizer.decode(response)
        action, action_input = split_action_action_input(response)
        if action is None:
            return None

        return [ChatCompletionMessageToolCall(function=Function(name=action, arguments=action_input))]

    @torch.inference_mode()
    async def infer_async(self,
                          template: Template,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = request_config or RequestConfig()

        inputs = template.encode(infer_request)
        assert len(inputs) >= 0
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template)
        infer_args = (template, inputs, generation_config)
        if request_config.stream:
            return self._infer_stream_async(*infer_args, **kwargs)
        else:
            return await self._infer_full_async(*infer_args, **kwargs)

    @staticmethod
    def _get_num_tokens(inputs: Dict[str, Any]) -> int:
        if 'input_ids' in inputs:  # 1d or 2d
            input_ids = inputs['input_ids']
            if isinstance(input_ids, list):
                return len(input_ids)
            else:
                return input_ids.shape[-1]
        elif 'inputs_embeds' in inputs:  # 2d or 3d
            return inputs['inputs_embeds'].shape[-1]
        raise ValueError(f'Unable to retrieve input_ids and inputs_embeds. inputs: {inputs}')

    def set_default_max_tokens(self,
                               request_config: RequestConfig,
                               inputs: Dict[str, Any],
                               strict: bool = False) -> None:
        max_model_len = self.max_model_len
        if isinstance(inputs, dict):
            inputs = [inputs]
        # The num_tokens takes the maximum value from inputs_list.
        num_tokens = 0
        for inp in inputs:
            num_tokens = max(num_tokens, self._get_num_tokens(inp))
        max_tokens = request_config.max_tokens
        if max_model_len is None:
            max_model_len = 8192
            logger.warning(
                'The current model is unable to retrieve `max_model_len`. It is set to the default value of 8192.')
        max_max_tokens = max_model_len - num_tokens
        if max_tokens is None:
            request_config.max_tokens = max_max_tokens
        elif max_max_tokens < request_config.max_tokens:
            if strict:
                raise ValueError(
                    f'Your prompt has {num_tokens} tokens, and you have set the `max_tokens` to {max_tokens}, '
                    f'but the maximum model length supported is {max_model_len}. '
                    'Please reduce the number of tokens in the prompt or the `max_tokens`.')
            else:
                logger.warning(f'max_model_len({max_model_len}) - num_tokens({num_tokens}) < max_tokens({max_tokens}). '
                               f'Setting max_tokens: {max_model_len - num_tokens}')
                request_config.max_tokens = max_max_tokens

    @staticmethod
    def _get_logprobs(tokenizer: PreTrainedTokenizerBase,
                      logprobs_list: Optional[List[Dict[int, float]]],
                      token_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if logprobs_list is None or len(token_ids) == 0:
            return None
        if len(token_ids) > 0:
            logprobs_list = logprobs_list[-len(token_ids):]
        res = []
        for logprobs, token_id in zip(logprobs_list, token_ids):
            token = tokenizer.decode(token_id)
            _res = {'token': token, 'logprob': logprobs[token_id], 'bytes': list(token.encode('utf8'))}
            if top_logprobs is not None:
                res_top_logprobs = []
                for k, logprob in logprobs.items():
                    if k == token_id:  # TODO
                        continue
                    token = tokenizer.decode(k)
                    res_top_logprobs.append({'token': token, 'logprob': logprob, 'bytes': list(token.encode('utf8'))})
                _res['top_logprobs'] = res_top_logprobs
            res.append(_res)
        return {'content': res}

    @staticmethod
    def _get_finish_reason(max_tokens: int, num_prompt_tokens: int, is_finished: bool):
        if is_finished:
            if num_prompt_tokens >= max_tokens:
                finish_reason = 'length'
            else:
                finish_reason = 'stop'
        else:
            finish_reason = None
        return finish_reason
