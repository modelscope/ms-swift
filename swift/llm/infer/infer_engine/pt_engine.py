# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import hashlib
import inspect
import pickle
import time
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

import json
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessorList
from transformers.utils import is_torch_npu_available

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer, safe_snapshot_download, to_device
from swift.plugin import Metric
from swift.tuners import Swift
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from .infer_engine import InferEngine
from .utils import AdapterRequest, InferStreamer, LogitsStreamer, TokensIteratorStreamer, prepare_generation_config


class _GenerationConfig(GenerationConfig):

    def __repr__(self) -> str:
        parameters = inspect.signature(self.to_json_string).parameters
        kwargs = {}
        if 'ignore_metadata' in parameters:
            kwargs['ignore_metadata'] = True
        gen_kwargs = json.loads(self.to_json_string(**kwargs))
        gen_kwargs.pop('transformers_version', None)
        return f'GenerationConfig({gen_kwargs})'


class PtEngine(InferEngine):

    def __init__(
            self,
            model_id_or_path: str,
            torch_dtype: Optional[torch.dtype] = None,
            *,
            adapters: List[str] = None,
            max_batch_size: int = 1,  # 0/1: no limit
            model_type: Optional[str] = None,
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            hub_token: Optional[str] = None,
            load_model: bool = True,
            # model kwargs
            attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
            device_map: Optional[Union[str, Dict[str, Any]]] = None,
            quantization_config=None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            template: Optional[Template] = None,
            **kwargs):
        self.model, self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_impl=attn_impl,
            model_kwargs=model_kwargs,
            **kwargs)
        self.max_batch_size = max_batch_size
        if isinstance(adapters, str):
            adapters = [adapters]
        self.adapters = adapters or []
        for adapter in self.adapters:
            self._add_adapter(safe_snapshot_download(adapter, use_hf=use_hf, hub_token=hub_token))
        self._post_init(template)

    def _post_init(self, template=None):
        super()._post_init(template)
        self.engine = self.model  # dummy
        self.generation_config = self.model.generation_config
        self._queue = Queue()
        self._task_pool = {}
        self._task_thread = None

    def _start_infer_worker(self):
        self._task_thread = Thread(target=self._infer_worker, daemon=True)
        self._task_thread.start()

    def _fetch_infer_requests(self):
        while not self._queue.empty():
            infer_request, kwargs, queue = self._queue.get()
            template = kwargs['template']
            info = hashlib.sha256(pickle.dumps((kwargs['request_config'], template
                                                and template.template_meta))).hexdigest()
            if info not in self._task_pool:
                self._task_pool[info] = kwargs, []
            self._task_pool[info][1].append((infer_request, queue))
        if len(self._task_pool) == 0:
            return
        key, (kwargs, data) = next(iter(self._task_pool.items()))
        max_batch_size = self.max_batch_size
        if max_batch_size <= 0:
            max_batch_size = len(data)
        data, remain_data = data[:max_batch_size], data[max_batch_size:]
        if remain_data:
            self._task_pool[key] = kwargs, remain_data
        else:
            self._task_pool.pop(key)
        kwargs = kwargs.copy()
        kwargs['infer_requests'] = [d[0] for d in data]
        queue_list = [d[1] for d in data]
        return kwargs, queue_list

    def _infer_worker(self):
        while True:
            time.sleep(0.01)
            item = self._fetch_infer_requests()
            if item is not None:
                kwargs, queue_list = item
                request_config = kwargs['request_config']
                res_list_or_gen = self._infer(**kwargs)
                if request_config.stream:
                    finished = False
                    while not finished:
                        try:
                            res_list = next(res_list_or_gen)
                        except StopIteration:
                            finished = True
                            res_list = [None] * len(queue_list)
                        for (queue, loop), res in zip(queue_list, res_list):
                            asyncio.run_coroutine_threadsafe(queue.put(res), loop)
                else:
                    for (queue, loop), res in zip(queue_list, res_list_or_gen):
                        asyncio.run_coroutine_threadsafe(queue.put(res), loop)

    def _add_adapter(self, adapter_path: str, adapter_name: Optional[str] = None) -> None:
        self.model = Swift.from_pretrained(self.model, adapter_path, adapter_name)

    @classmethod
    def from_model_template(cls, model, template=None, *, max_batch_size: int = 1):
        self = super().__new__(cls)
        self.model = model
        self.processor = template.processor
        self.max_batch_size = max_batch_size
        self._post_init(template)
        return self

    def _prepare_generation_config(self, request_config: RequestConfig) -> _GenerationConfig:
        generation_config = prepare_generation_config(self.generation_config, request_config, self.tokenizer)
        generation_config.return_dict_in_generate = True
        if request_config.logprobs:
            generation_config.output_logits = True
        generation_config.top_logprobs = request_config.top_logprobs
        generation_config.num_return_sequences = request_config.n
        return _GenerationConfig(**generation_config.to_dict())

    def _add_stop_words(self, generation_config: _GenerationConfig, request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:
        stop_words = (request_config.stop or []) + template_meta.stop_words
        generation_config.stop_words = self._get_stop_words(stop_words)

    @staticmethod
    def preprocess_logits(batched_logits: Optional[List[torch.Tensor]], batched_generate_ids: torch.Tensor,
                          top_logprobs: Optional[int]):
        top_logprobs = top_logprobs or 1
        batch_size = batched_generate_ids.shape[0]
        if batched_logits is None:
            return None
        batched_logprobs = []
        for i in range(batch_size):
            logprobs_list = []
            generate_ids = batched_generate_ids[i]
            for j, logits in enumerate(batched_logits):
                token = generate_ids[j].item()
                logprobs = torch.log_softmax(logits[i], -1)
                tokens = [token] + logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
                logprobs_list.append({token: logprobs[token].item() for token in tokens})
            batched_logprobs.append(logprobs_list)
        return batched_logprobs

    @staticmethod
    def _update_batched_logprobs(batched_logprobs: List[torch.Tensor], logits_streamer: Optional[LogitsStreamer],
                                 generate_ids: torch.Tensor, top_logprobs: int) -> None:
        seq_len = generate_ids.shape[1] - len(batched_logprobs[0])
        if logits_streamer is None or seq_len == 0:
            return

        res = []
        for i in range(seq_len):
            res.append(logits_streamer.queue.get())
        new_batched_logprobs = PtEngine.preprocess_logits(res, generate_ids[:, -seq_len:], top_logprobs)
        for logprobs, new_logprobs in zip(batched_logprobs, new_batched_logprobs):
            logprobs += new_logprobs

    def _infer_stream(self,
                      template: Template,
                      inputs: Dict[str, Any],
                      *,
                      generation_config: GenerationConfig,
                      adapter_request: Optional[AdapterRequest] = None,
                      **kwargs) -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:

        if generation_config.num_beams != 1:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)
        streamer = TokensIteratorStreamer()
        generate_kwargs = {
            'generation_config': generation_config,
            'streamer': streamer,
            **inputs,
        }
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            generate_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)

        logits_streamer = None
        if generation_config.output_logits:
            generate_kwargs['logits_processor'] = LogitsProcessorList([LogitsStreamer()])

        def _model_generate(**kwargs):
            if is_torch_npu_available():
                torch.npu.set_device(self.model.device)
            template.generate(self.model, **kwargs)

        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        thread = Thread(target=_model_generate, kwargs=generate_kwargs)
        thread.start()
        batch_size = inputs['attention_mask'].shape[0]
        all_is_finished = False
        is_finished = [False] * batch_size
        infer_streamers = [InferStreamer(template) for _ in range(batch_size)]
        request_id_list = [f'chatcmpl-{random_uuid()}' for _ in range(batch_size)]
        token_idxs = [0] * batch_size

        raw_batched_generate_ids = None  # or torch.Tensor: [batch_size, seq_len]
        batched_logprobs = [[] for _ in range(batch_size)]
        while not all_is_finished:
            try:
                batched_tokens = next(streamer)
                if batched_tokens.ndim == 1:
                    batched_tokens = batched_tokens[:, None]

                raw_batched_generate_ids = torch.concat(
                    [batched_tokens]
                    if raw_batched_generate_ids is None else [raw_batched_generate_ids, batched_tokens],
                    dim=1)
            except StopIteration:
                all_is_finished = True

            batched_generate_ids = template.get_generate_ids(raw_batched_generate_ids, num_prompt_tokens)
            self._update_batched_logprobs(batched_logprobs, logits_streamer, batched_generate_ids,
                                          generation_config.top_logprobs)

            res = []
            for i in range(batched_generate_ids.shape[0]):
                if is_finished[i]:
                    res.append(None)
                    continue
                generate_ids = batched_generate_ids[i]

                # ignore pad_token
                masks = generate_ids != self.tokenizer.pad_token_id
                generate_ids = generate_ids[masks].tolist()
                logprobs_list = None
                if batched_logprobs[i]:
                    logprobs_list = [logprobs for m, logprobs in zip(masks, batched_logprobs[i]) if m.item()]

                is_finished[i] = (
                    all_is_finished or is_finished[i]
                    or len(generate_ids) > 0 and generate_ids[-1] == self.tokenizer.pad_token_id)
                delta_text = infer_streamers[i].get_printable_text(generate_ids, is_finished[i])
                if not delta_text and not is_finished[i]:
                    res.append(None)
                    continue
                logprobs = self._get_logprobs(logprobs_list, generate_ids[token_idxs[i]:],
                                              generation_config.top_logprobs)
                token_idxs[i] = len(generate_ids)

                usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))
                toolcall = None
                if is_finished[i]:
                    toolcall = self._get_toolcall(template.decode(generate_ids), template)
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, num_prompt_tokens,
                                                        is_finished[i])

                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs)
                ]
                res.append(
                    ChatCompletionStreamResponse(
                        model=self.model_name, choices=choices, usage=usage_info, id=request_id_list[i]))
            if any(res):
                yield res

    def _get_adapter_names(self, adapter_request: Optional[AdapterRequest]) -> Optional[List[str]]:
        if adapter_request is None:
            if self._adapters_pool:
                return ['__base__']
            return
        adapter_name = adapter_request.name
        if adapter_name not in self._adapters_pool:
            self._adapters_pool[adapter_name] = adapter_request
            self._add_adapter(adapter_request.path, adapter_name)
        return [adapter_name]

    def _infer_forward(self,
                       template: Template,
                       inputs: Dict[str, Any],
                       adapter_request: Optional[AdapterRequest] = None,
                       **kwargs):
        call_kwargs = {}
        top_logprobs = getattr(kwargs.get('generation_config'), 'top_logprobs', None) or 20
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            call_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)
        inputs.pop('labels', None)
        logits = self.model(**inputs, **call_kwargs).logits
        if template.mode == 'seq_cls':
            preds, logprobs = template.decode_seq_cls(logits, top_logprobs)
        elif template.mode == 'prm':
            preds = template.decode_prm(inputs['input_ids'], logits)
            logprobs = [None] * len(preds)
        else:
            raise ValueError(f'Unsupported mode: {template.mode}')

        res = []
        for i, pred in enumerate(preds):
            usage_info = self._get_usage_info(num_prompt_tokens, 1)
            choices = [
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role='assistant', content=pred, tool_calls=None),
                    finish_reason='stop',
                    logprobs=logprobs[i])
            ]
            res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))
        return res

    def _infer_full(self,
                    template: Template,
                    inputs: Dict[str, Any],
                    *,
                    generation_config: GenerationConfig,
                    adapter_request: Optional[AdapterRequest] = None,
                    template_inputs=None) -> List[ChatCompletionResponse]:
        # bos_token TODO: encoder-decoder
        generate_kwargs = {'generation_config': generation_config, **inputs}
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            generate_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)
        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        output = dict(template.generate(self.model, **generate_kwargs))
        output.pop('past_key_values', None)
        batched_generate_ids = output['sequences']
        batched_generate_ids = template.get_generate_ids(batched_generate_ids, num_prompt_tokens)
        template.debug_logger({'generate_ids': batched_generate_ids})  # debug
        batched_logprobs = self.preprocess_logits(
            output.get('logits'), batched_generate_ids, generation_config.top_logprobs)

        res = []
        num_return_sequences = generation_config.num_return_sequences
        for i in range(inputs['attention_mask'].shape[0]):
            choices = []
            usage_info = self._get_usage_info(num_prompt_tokens, 0)
            for j in range(num_return_sequences):
                batched_index = i * num_return_sequences + j
                generate_ids = batched_generate_ids[batched_index]

                # ignore pad_token
                masks = generate_ids != self.tokenizer.pad_token_id
                generate_ids = generate_ids[masks].tolist()
                logprobs_list = None
                if batched_logprobs is not None:
                    logprobs_list = [
                        logprobs for m, logprobs in zip(masks, batched_logprobs[batched_index]) if m.item()
                    ]

                logprobs = self._get_logprobs(logprobs_list, generate_ids, generation_config.top_logprobs)
                usage_info = self._update_usage_info(usage_info, len(generate_ids))
                response = template.decode(generate_ids, template_inputs=template_inputs[i])
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, num_prompt_tokens, True)
                toolcall = self._get_toolcall(response, template)
                choices.append(
                    ChatCompletionResponseChoice(
                        index=j,
                        message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs))
            res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))
        return res

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        if request_config is None:
            request_config = RequestConfig()
        queue = asyncio.Queue()
        self._queue.put((infer_request, {
            'request_config': request_config,
            'template': template,
            'adapter_request': adapter_request,
            'pre_infer_hook': pre_infer_hook
        }, (queue, asyncio.get_event_loop())))
        await asyncio.sleep(0)
        if self._task_thread is None:
            self._start_infer_worker()
        if request_config.stream:

            async def _gen_wrapper():
                while True:
                    item = await queue.get()
                    await asyncio.sleep(0)
                    if item is None:
                        break
                    yield item

            return _gen_wrapper()
        else:
            return await queue.get()

    @staticmethod
    def _add_error_list(outputs, error_list):
        for i, error in error_list:
            outputs.insert(i, error)
        return outputs

    # Ensure `template._post_encode` has no gradient.
    @torch.inference_mode()
    def _infer(
        self,
        infer_requests: List[InferRequest],
        request_config: RequestConfig,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        self.model.eval()
        request_config = deepcopy(request_config)
        if template is None:
            template = self.default_template
        if template.use_model:
            template.model = self.model

        generation_config = None
        if self.model_info.task_type == 'causal_lm':
            template.set_mode('pt')

        batched_inputs, error_list = self._batch_encode(
            infer_requests, template=template, strict=getattr(self, 'strict', True))
        if len(batched_inputs) > 0:
            template_inputs = [inputs.pop('template_inputs') for inputs in batched_inputs]
            inputs = to_device(template.data_collator(batched_inputs), self.model.device)
            template.debug_logger(inputs)  # debug
            if self.model.model_meta.is_multimodal:
                _, inputs = template.pre_forward_hook(self.model, None, inputs)
            if self.model_info.task_type == 'causal_lm':
                self.set_default_max_tokens(request_config, inputs)
                generation_config = self._prepare_generation_config(request_config)
                self._add_stop_words(generation_config, request_config, template.template_meta)
            else:
                generation_config = request_config

            kwargs = {
                'template': template,
                'inputs': inputs,
                'generation_config': generation_config,
                'adapter_request': adapter_request,
                'template_inputs': template_inputs
            }
            if pre_infer_hook:
                kwargs = pre_infer_hook(kwargs)
        else:
            kwargs = {}
        if request_config.stream:

            def _gen_wrapper():
                if len(kwargs) > 0:
                    for res in self._infer_stream(**kwargs):
                        yield self._add_error_list(res, error_list)
                else:
                    yield self._add_error_list([], error_list)

            return _gen_wrapper()
        else:
            if len(kwargs) > 0:
                infer_func = self._infer_forward if template.mode in ('seq_cls', 'prm') else self._infer_full
                res = infer_func(**kwargs)
            else:
                res = []
            return self._add_error_list(res, error_list)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        if request_config is None:
            request_config = RequestConfig()
        if request_config.stream:
            return super().infer(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request)
        # Has higher stability than calling super().infer
        if use_tqdm is None:
            use_tqdm = not request_config.stream and len(infer_requests) > 1
        prog_bar = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)
        # If self.max_batch_size <= 0, then process all infer_requests at once.
        max_batch_size = self.max_batch_size
        if max_batch_size <= 0:
            max_batch_size = len(infer_requests)
        res = []
        i = 0
        while i < len(infer_requests):
            infer_requests_samples = infer_requests[i:i + max_batch_size]
            res += self._infer(
                infer_requests_samples, request_config, template=template, adapter_request=adapter_request)
            i += max_batch_size
            prog_bar.update(len(infer_requests_samples))
        self._update_metrics(res, metrics)
        return res
