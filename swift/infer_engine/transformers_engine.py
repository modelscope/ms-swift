# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import hashlib
import inspect
import pickle
import time
from copy import deepcopy
from queue import Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessorList
from transformers.utils import is_torch_npu_available

from swift.metrics import Metric
from swift.model import get_model_processor
from swift.template import Template
from swift.tuners import Swift
from swift.utils import get_generative_reranker_logits, safe_snapshot_download, to_device
from .infer_engine import InferEngine
from .protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                       ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,
                       EmbeddingResponseData, InferRequest, RequestConfig, random_uuid)
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


class TransformersEngine(InferEngine):

    def __init__(
            self,
            model: Union[str, nn.Module],
            *,
            template: Optional[Template] = None,
            adapters: Optional[List[str]] = None,
            max_batch_size: int = 1,  # 0/1: no limit
            reranker_use_activation: bool = True,
            # model kwargs
            torch_dtype: Optional[torch.dtype] = None,
            model_type: Optional[str] = None,
            attn_impl: Optional[str] = None,
            device_map: Optional[Union[str, Dict[str, Any]]] = None,
            task_type: Optional[str] = None,
            quantization_config=None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            # hub kwargs
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            hub_token: Optional[str] = None,
            **kwargs):
        if isinstance(adapters, str):
            adapters = [adapters]
        self.adapters = adapters or []
        self.max_batch_size = max_batch_size
        self.reranker_use_activation = reranker_use_activation

        self.torch_dtype = torch_dtype
        self.model_type = model_type
        self.attn_impl = attn_impl
        self.device_map = device_map
        self.task_type = task_type
        self.quantization_config = quantization_config
        self.model_kwargs = model_kwargs

        self.use_hf = use_hf
        self.revision = revision
        self.hub_token = hub_token
        if isinstance(model, str):
            self.model, processor = self._get_model_processor(model, **kwargs)
            template = self._get_template(processor)
        elif isinstance(model, nn.Module):
            self.model = model
            if template is None:
                raise ValueError('`template` is required when `model` is a nn.Module')
        super().__init__(template)
        for adapter in self.adapters:
            self._add_adapter(safe_snapshot_download(adapter, use_hf=self.use_hf, hub_token=self.hub_token))
        self.engine = self.model  # dummy
        self.generation_config = getattr(self.model, 'generation_config', None)
        self._queue = Queue()
        self._task_pool = {}
        self._adapters_pool = {}
        self._task_thread = None

    def _get_model_processor(self, model_id_or_path, **kwargs):
        return get_model_processor(
            model_id_or_path,
            torch_dtype=self.torch_dtype,
            model_type=self.model_type,
            use_hf=self.use_hf,
            hub_token=self.hub_token,
            revision=self.revision,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            attn_impl=self.attn_impl,
            task_type=self.task_type,
            model_kwargs=self.model_kwargs,
            **kwargs)

    def _start_infer_worker(self):
        self._task_thread = Thread(target=self._infer_worker, daemon=True)
        self._task_thread.start()

    def _fetch_infer_requests(self):
        while not self._queue.empty():
            infer_request, kwargs, queue = self._queue.get()
            info = hashlib.sha256(pickle.dumps((kwargs['request_config']))).hexdigest()
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

    def _prepare_generation_config(self, request_config: RequestConfig) -> _GenerationConfig:
        generation_config = prepare_generation_config(self.generation_config, request_config, self.tokenizer)
        generation_config.return_dict_in_generate = True
        if request_config.logprobs:
            generation_config.output_logits = True
        generation_config.num_return_sequences = request_config.n
        return _GenerationConfig(**generation_config.to_dict())

    def _add_stop_words(self, generation_config: _GenerationConfig, request_config: RequestConfig) -> None:
        template_meta = self.template.template_meta
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
        new_batched_logprobs = TransformersEngine.preprocess_logits(res, generate_ids[:, -seq_len:], top_logprobs)
        for logprobs, new_logprobs in zip(batched_logprobs, new_batched_logprobs):
            logprobs += new_logprobs

    def _infer_stream(self, inputs: Dict[str, Any], *, generation_config: GenerationConfig,
                      adapter_request: Optional[AdapterRequest], request_config: RequestConfig,
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
            self.template.generate(self.model, **kwargs)

        generate_kwargs = self.template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        thread = Thread(target=_model_generate, kwargs=generate_kwargs)
        thread.start()
        batch_size = inputs['attention_mask'].shape[0]
        all_is_finished = False
        is_finished = [False] * batch_size
        infer_streamers = [InferStreamer(self.template) for _ in range(batch_size)]
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

            batched_generate_ids = self.template.get_generate_ids(raw_batched_generate_ids, num_prompt_tokens)
            self._update_batched_logprobs(batched_logprobs, logits_streamer, batched_generate_ids,
                                          request_config.top_logprobs)

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
                logprobs = self._get_logprobs(logprobs_list, generate_ids[token_idxs[i]:], request_config.top_logprobs)
                token_idxs[i] = len(generate_ids)

                usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))
                toolcall = None
                if is_finished[i]:
                    toolcall = self._get_toolcall(self.template.decode(generate_ids))
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, usage_info.completion_tokens,
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

    def _infer_forward(self, inputs: Dict[str, Any], adapter_request: Optional[AdapterRequest],
                       request_config: RequestConfig, **kwargs):
        call_kwargs = {}
        top_logprobs = request_config.top_logprobs or 20
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            call_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)
        inputs.pop('labels', None)
        output = self.model(**inputs, **call_kwargs)
        if hasattr(output, 'logits'):
            logits = output.logits
        elif 'last_hidden_state' in output:
            # embeddings
            logits = output['last_hidden_state']
        else:
            raise NotImplementedError('Only support `logits` or `hidden_state` in output.')
        task_type = self.template.task_type
        if task_type == 'seq_cls':
            preds, logprobs = self.template.decode_seq_cls(logits, top_logprobs)
        elif task_type == 'prm':
            preds = self.template.decode_prm(inputs['input_ids'], logits)
            logprobs = [None] * len(preds)
        elif task_type == 'embedding':
            preds = logits
            logprobs = [None] * len(preds)
        elif task_type in ('reranker', 'generative_reranker'):
            if task_type == 'generative_reranker':
                # Qwen3-reranker like
                logits = get_generative_reranker_logits(
                    self.template.tokenizer, logits, attention_mask=inputs.get('attention_mask'))
            preds = logits.float()
            if self.reranker_use_activation:
                preds = F.sigmoid(preds)
            preds = preds.tolist()
            logprobs = [None] * len(preds)
        else:
            raise ValueError(f'Unsupported task_type: {task_type}')

        res = []
        for i, pred in enumerate(preds):
            usage_info = self._get_usage_info(num_prompt_tokens, 1)
            if task_type == 'embedding':
                res.append(
                    EmbeddingResponse(
                        model=self.model_name, usage=usage_info, data=[EmbeddingResponseData(embedding=pred.tolist())]))
            else:
                choices = [
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role='assistant', content=pred, tool_calls=None),
                        finish_reason='stop',
                        logprobs=logprobs[i])
                ]
                res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))
        return res

    def _infer_full(self, inputs: Dict[str, Any], *, generation_config: GenerationConfig,
                    adapter_request: Optional[AdapterRequest], request_config: RequestConfig,
                    template_inputs) -> List[ChatCompletionResponse]:
        # bos_token TODO: encoder-decoder
        generate_kwargs = {'generation_config': generation_config, **inputs}
        adapter_names = self._get_adapter_names(adapter_request)
        if adapter_names is not None:
            generate_kwargs['adapter_names'] = adapter_names
        num_prompt_tokens = self._get_num_tokens(inputs)
        generate_kwargs = self.template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        output = dict(self.template.generate(self.model, **generate_kwargs))
        output.pop('past_key_values', None)
        batched_generate_ids = output['sequences']
        batched_generate_ids = self.template.get_generate_ids(batched_generate_ids, num_prompt_tokens)
        self.template.debug_logger({'generate_ids': batched_generate_ids})  # debug
        batched_logprobs = self.preprocess_logits(
            output.get('logits'), batched_generate_ids, request_config.top_logprobs)

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

                logprobs = self._get_logprobs(logprobs_list, generate_ids, request_config.top_logprobs)
                usage_info = self._update_usage_info(usage_info, len(generate_ids))
                response = self.template.decode(generate_ids, template_inputs=template_inputs[i])
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, len(generate_ids), True)
                toolcall = self._get_toolcall(response)
                token_ids = generate_ids if request_config.return_details else None
                choices.append(
                    ChatCompletionResponseChoice(
                        index=j,
                        message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                        token_ids=token_ids))
            prompt_token_ids = None
            images_size = None
            if request_config.return_details:
                if 'input_ids' in inputs:
                    non_pad_indices = (inputs['input_ids'][i] != self.tokenizer.pad_token_id).nonzero()
                    if non_pad_indices.numel() > 0:
                        idx = non_pad_indices.min().item()
                        prompt_token_ids = inputs['input_ids'][i][idx:].tolist()
                if all(isinstance(image, Image.Image) for image in template_inputs[i].images):
                    images_size = [image.size for image in template_inputs[i].images]
            res.append(
                ChatCompletionResponse(
                    model=self.model_name,
                    choices=choices,
                    usage=usage_info,
                    prompt_token_ids=prompt_token_ids,
                    images_size=images_size))
        return res

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        if request_config is None:
            request_config = RequestConfig()
        queue = asyncio.Queue()
        self._queue.put((infer_request, {
            'request_config': request_config,
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

    # Ensure `template._post_encode` has no gradient.
    @torch.inference_mode()
    def _infer(
        self,
        infer_requests: List[InferRequest],
        request_config: RequestConfig,
        *,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        self.model.eval()
        request_config = deepcopy(request_config)
        if self.template.use_model:
            self.template.model = self.model

        if self.model_info.task_type == 'causal_lm':
            self.template.set_mode('pt')

        batched_inputs, error_list = self._batch_encode(infer_requests, strict=getattr(self, 'strict', True))
        if len(batched_inputs) > 0:
            template_inputs = [inputs.pop('template_inputs') for inputs in batched_inputs]
            inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
            self.template.debug_logger(inputs)  # debug
            if self.model_meta.is_multimodal:
                _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
            if self.model_info.task_type == 'causal_lm':
                self.set_default_max_tokens(request_config, inputs)
                generation_config = self._prepare_generation_config(request_config)
                self._add_stop_words(generation_config, request_config)
            else:
                generation_config = request_config

            kwargs = {
                'inputs': inputs,
                'generation_config': generation_config,
                'adapter_request': adapter_request,
                'request_config': request_config,
                'template_inputs': template_inputs,
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
                infer_func = self._infer_forward if self.template.task_type in {
                    'seq_cls', 'prm', 'embedding', 'reranker', 'generative_reranker'
                } else self._infer_full
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
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        if request_config is None:
            request_config = RequestConfig()
        if request_config.stream:
            return super().infer(
                infer_requests, request_config, metrics, use_tqdm=use_tqdm, adapter_request=adapter_request)
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
            res += self._infer(infer_requests_samples, request_config, adapter_request=adapter_request)
            i += max_batch_size
            prog_bar.update(len(infer_requests_samples))
        prog_bar.close()
        self._update_metrics(res, metrics)
        return res
