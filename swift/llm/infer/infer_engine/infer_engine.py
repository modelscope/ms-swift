# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import concurrent.futures
import os
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Union

from tqdm import tqdm

from swift.llm import InferRequest, ProcessorMixin, get_template
from swift.llm.template import Template, split_action_action_input
from swift.llm.utils import get_ckpt_dir
from swift.plugin import Metric
from swift.utils import get_logger
from ..protocol import (ChatCompletionMessageToolCall, ChatCompletionResponse, ChatCompletionStreamResponse, Function,
                        RequestConfig, UsageInfo)
from .base import BaseInferEngine

logger = get_logger()


class InferEngine(BaseInferEngine, ProcessorMixin):
    llm_max_batch_size = 1024 * 1024
    mllm_max_batch_size = 1024

    def _post_init(self):
        processor = self.processor
        self.model_info = processor.model_info
        self.model_meta = processor.model_meta
        self.model_dir = self.model_info.model_dir
        self.model_name = self.model_info.model_name
        self.max_model_len = self.model_info.max_model_len
        self.config = self.model_info.config
        if getattr(self, 'default_template', None) is None:
            ckpt_dir = get_ckpt_dir(self.model_dir, getattr(self, 'adapters', None))
            if ckpt_dir:
                from swift.llm import BaseArguments
                args = BaseArguments.from_pretrained(ckpt_dir)
                self.default_template = get_template(args.template, self.processor, default_system=args.system)
            else:
                self.default_template = get_template(self.model_meta.template, self.processor)

        self._adapters_pool = {}

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

    def async_iter_to_iter(self, async_iter, prog_bar, metrics) -> Iterator:
        queue = Queue()

        async def _run_async_iter():
            try:
                async for item in await async_iter:
                    queue.put(item)
            except Exception as e:
                if getattr(self, 'strict', True):
                    raise
                queue.put(e)
            else:
                queue.put(None)

        thread = Thread(target=lambda: asyncio.run(_run_async_iter()))
        thread.start()
        pre_output = None
        while True:
            output = queue.get()
            if output is None or isinstance(output, Exception):
                prog_bar.update()
                self._update_metrics(pre_output, metrics)
                return
            pre_output = output
            yield output

    @staticmethod
    async def batch_run(tasks):
        return await asyncio.gather(*tasks)

    def _batch_infer_stream(
        self,
        tasks,
        stream: bool = True,
        use_tqdm: bool = True,
        metrics: Optional[List[Metric]] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:

        prog_bar = tqdm(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)
        if stream:
            return [self.async_iter_to_iter(task, prog_bar, metrics) for task in tasks]
        else:

            async def _new_run(task):
                try:
                    res = await task
                except Exception as e:
                    if getattr(self, 'strict', True):
                        raise
                    res = e
                prog_bar.update()
                self._update_metrics(res, metrics)
                return res

            new_tasks = [_new_run(task) for task in tasks]
            return self.safe_asyncio_run(self.batch_run(new_tasks))

    @staticmethod
    def _get_usage_info(num_prompt_tokens: int, num_generated_tokens: int) -> UsageInfo:
        return UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

    @staticmethod
    def _update_usage_info(origin_use_info: UsageInfo, num_generated_tokens: int) -> UsageInfo:
        return UsageInfo(
            prompt_tokens=origin_use_info.prompt_tokens,
            completion_tokens=origin_use_info.completion_tokens + num_generated_tokens,
            total_tokens=origin_use_info.total_tokens + num_generated_tokens,
        )

    @staticmethod
    def _update_metrics(result, metrics: Optional[List[Metric]] = None):
        if metrics is None:
            return result
        result_origin = result
        if not isinstance(result, (list, tuple)):
            result = [result]
        for response in result:
            if response is None or isinstance(response, Exception):
                continue
            for metric in metrics:
                metric.update(response)
        return result_origin

    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        if request_config is None:
            request_config = RequestConfig()
        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = not request_config.stream and len(infer_requests) > 1
        if request_config.stream:
            return self._batch_infer_stream(tasks, True, use_tqdm, metrics)
        else:
            i = 0
            result = []
            max_batch_size = self.llm_max_batch_size
            if hasattr(self, 'model_meta') and self.model_meta.is_multimodal:
                # vllm & lmdeploy
                max_batch_size = self.mllm_max_batch_size
            prog_bar = tqdm(
                total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm or len(tasks) <= max_batch_size)
            while i < len(tasks):
                tasks_samples = tasks[i:i + max_batch_size]
                res = self._batch_infer_stream(tasks_samples, False, use_tqdm, metrics)
                result += res
                i += max_batch_size
                prog_bar.update(len(tasks_samples))
            return result

    @staticmethod
    def _get_toolcall(response: Union[str, List[Dict[str, Any]]],
                      tools_prompt='react_en') -> Optional[List[ChatCompletionMessageToolCall]]:
        if not isinstance(response, str):
            response = '\n'.join([resp['text'] for resp in response if resp['type'] == 'text'])

        action, action_input = split_action_action_input(response, tools_prompt=tools_prompt)
        if action is None:
            return None

        return [ChatCompletionMessageToolCall(function=Function(name=action, arguments=action_input))]

    @staticmethod
    def _get_num_tokens(inputs: Dict[str, Any]) -> int:
        if 'input_ids' in inputs:  # 1d or 2d
            input_ids = inputs['input_ids']
            if isinstance(input_ids, list):
                return len(input_ids)
            else:
                return input_ids.shape[-1]
        elif 'inputs_embeds' in inputs:  # 2d or 3d
            return inputs['inputs_embeds'].shape[-2]
        raise ValueError(f'Unable to retrieve input_ids and inputs_embeds. inputs: {inputs}')

    def set_default_max_tokens(self, request_config: RequestConfig, inputs: Dict[str, Any]) -> None:
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
            logger.warning(f'max_model_len({max_model_len}) - num_tokens({num_tokens}) < max_tokens({max_tokens}). '
                           f'Setting max_tokens: {max_model_len - num_tokens}')
            request_config.max_tokens = max_max_tokens

    def _get_logprobs(self,
                      logprobs_list: Optional[List[Dict[int, float]]],
                      token_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if logprobs_list is None or len(token_ids) == 0:
            return None
        if len(token_ids) > 0:
            logprobs_list = logprobs_list[-len(token_ids):]
        res = []
        for logprobs, token_id in zip(logprobs_list, token_ids):
            token = self.tokenizer.decode(token_id)
            _res = {'token': token, 'logprob': logprobs[token_id], 'bytes': list(token.encode('utf8'))}
            if top_logprobs is not None:
                logprobs = {k: logprobs[k] for k in sorted(logprobs, key=lambda k: -logprobs[k])[:top_logprobs]}
                res_top_logprobs = []
                for k, logprob in logprobs.items():
                    if logprob == float('-inf'):
                        continue
                    token = self.tokenizer.decode(k)
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

    @staticmethod
    def thread_run(target, args=(), kwargs=None):
        kwargs = kwargs or {}

        def func(target, queue, args, kwargs):
            try:
                queue.put(target(*args, **kwargs))
            except Exception as e:
                queue.put(e)

        queue = Queue()
        thread = Thread(target=func, args=(target, queue, args, kwargs))
        thread.start()
        thread.join()
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def safe_asyncio_run(coro):
        return InferEngine.thread_run(asyncio.run, args=(coro, ))

    @staticmethod
    def _batch_encode(infer_requests: List[InferRequest], template: Template, strict: bool):
        max_workers = max(min(32, os.cpu_count(), len(infer_requests)), 1)
        error_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(template.encode, infer_request, return_template_inputs=True)
                for infer_request in infer_requests
            ]
            concurrent.futures.wait(futures)
            batched_inputs = []
            for i, future in enumerate(futures):
                try:
                    batched_inputs.append(future.result())
                except Exception as e:
                    if strict:
                        raise
                    error_list.append((i, e))
                    continue
        return batched_inputs, error_list
