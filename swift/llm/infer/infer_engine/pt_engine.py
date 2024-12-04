# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
from copy import deepcopy
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

import json
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LogitsProcessorList
from transformers.utils import is_torch_npu_available

from swift.llm import InferRequest, Template, get_model_tokenizer, to_device
from swift.plugin import Metric
from swift.tuners import Swift
from swift.utils import get_logger
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
from .infer_engine import InferEngine
from .utils import InferStreamer, LogitsStreamer, LoRARequest, TokensIteratorStreamer, prepare_generation_config

logger = get_logger()


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
            max_batch_size: int = 1,
            # hub kwargs
            model_type: Optional[str] = None,
            use_hf: Optional[bool] = None,
            revision: Optional[str] = None,
            load_model: bool = True,
            # model kwargs
            attn_impl: Literal['flash_attn', 'sdpa', 'eager', None] = None,
            device_map: Optional[Union[str, Dict[str, Any]]] = None,
            quantization_config: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs):
        self.model, self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            revision=revision,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_impl=attn_impl,
            model_kwargs=model_kwargs,
            **kwargs)
        self.max_batch_size = max_batch_size
        self._post_init()

    def _post_init(self):
        super()._post_init()
        self.engine = self.model  # dummy
        self.generation_config = self.model.generation_config
        self._lora_request_pool = {}

    @classmethod
    def from_model_processor(cls, model, processor, *, max_batch_size: int = 1):
        self = super().__new__(cls)
        self.model = model
        self.processor = processor
        self.max_batch_size = max_batch_size
        self._post_init()
        return self

    def _prepare_generation_config(self, request_config: RequestConfig) -> _GenerationConfig:
        generation_config = prepare_generation_config(self.generation_config, request_config, self.tokenizer)
        generation_config.return_dict_in_generate = True
        if request_config.logprobs:
            generation_config.output_logits = True
        generation_config.top_logprobs = request_config.top_logprobs
        return _GenerationConfig(**generation_config.to_dict())

    def _add_stop_words(self, generation_config: _GenerationConfig, request_config: RequestConfig,
                        template: Template) -> None:
        template_meta = template.template_meta
        stop_words = (request_config.stop or []) + template_meta.stop_words
        generation_config.stop_words = self._get_stop_words(stop_words)

    @staticmethod
    def preprocess_logits(batched_logits: Optional[List[torch.Tensor]], batched_generate_ids: torch.Tensor,
                          top_logprobs: int):
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
            return batched_logprobs

        res = []
        for i in range(seq_len):
            res.append(logits_streamer.queue.get())
        new_batched_logprobs = PtEngine.preprocess_logits(res, generate_ids[:, -seq_len:], top_logprobs)
        for logprobs, new_logprobs in zip(batched_logprobs, new_batched_logprobs):
            logprobs += new_logprobs

    def _infer_stream(self,
                      template: Template,
                      inputs: Dict[str, Any],
                      generation_config: GenerationConfig,
                      *,
                      lora_request: Optional[LoRARequest] = None,
                      **kwargs) -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:
        generate_kwargs = {}
        if lora_request is not None:
            generate_kwargs['adapter_names'] = self._get_adapter_names(lora_request)
        if generation_config.num_beams != 1:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)
        num_prompt_tokens = self._get_num_tokens(inputs)

        streamer = TokensIteratorStreamer()
        generate_kwargs.update({
            'generation_config': generation_config,
            'streamer': streamer,
            **inputs,
        })
        logits_streamer = None
        if generation_config.output_logits:
            generate_kwargs['logits_processor'] = LogitsProcessorList([LogitsStreamer()])

        def _model_generate(*args, **kwargs):
            if is_torch_npu_available():
                torch.npu.set_device(self.model.device)
            self.model.generate(*args, **kwargs)

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
                                          generation_config.top_logprobs or 1)
            # TODO: MLLM
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
                logprobs = self._get_logprobs(self.tokenizer, logprobs_list, generate_ids[token_idxs[i]:],
                                              generation_config.top_logprobs)
                token_idxs[i] = len(generate_ids)

                usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))
                toolcall = None
                if is_finished[i]:
                    toolcall = self._get_toolcall(template.decode(generate_ids))
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
                        model=self.model_dir, choices=choices, usage=usage_info, id=request_id_list[i]))
            if any(res):
                yield res

    def _get_adapter_names(self, lora_request: LoRARequest) -> str:
        lora_name = lora_request.lora_name
        lora_path = lora_request.lora_path
        if lora_name in self._lora_request_pool:
            assert lora_path == self._lora_request_pool[lora_name].lora_path
        else:
            self._lora_request_pool[lora_name] = lora_request
            self.model = Swift.from_pretrained(self.model, lora_path, lora_name, inference_mode=True)
        return [lora_name]

    def _infer_full(self,
                    template: Template,
                    inputs: Dict[str, Any],
                    generation_config: GenerationConfig,
                    *,
                    lora_request: Optional[LoRARequest] = None,
                    template_inputs=None) -> Union[List[ChatCompletionResponse]]:
        # bos_token TODO: encoder-decoder
        generate_kwargs = {}
        if lora_request is not None:
            generate_kwargs['adapter_names'] = self._get_adapter_names(lora_request)
        generate_kwargs.update({'generation_config': generation_config, **inputs})
        num_prompt_tokens = self._get_num_tokens(inputs)

        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)
        output = dict(self.model.generate(**generate_kwargs))
        output.pop('past_key_values', None)
        batched_generate_ids = output['sequences']
        batched_generate_ids = template.get_generate_ids(batched_generate_ids, num_prompt_tokens)
        batched_logprobs = self.preprocess_logits(
            output.get('logits'), batched_generate_ids, generation_config.top_logprobs)

        res = []
        for i in range(batched_generate_ids.shape[0]):
            generate_ids = batched_generate_ids[i]

            # ignore pad_token
            masks = generate_ids != self.tokenizer.pad_token_id
            generate_ids = generate_ids[masks].tolist()
            logprobs_list = None
            if batched_logprobs is not None:
                logprobs_list = [logprobs for m, logprobs in zip(masks, batched_logprobs[i]) if m.item()]

            logprobs = self._get_logprobs(self.tokenizer, logprobs_list, generate_ids, generation_config.top_logprobs)
            usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))
            response = template.decode(generate_ids, template_inputs=template_inputs[i])
            finish_reason = self._get_finish_reason(generation_config.max_new_tokens, num_prompt_tokens, True)
            toolcall = self._get_toolcall(response)
            choices = [
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                    finish_reason=finish_reason,
                    logprobs=logprobs)
            ]
            res.append(ChatCompletionResponse(model=self.model_dir, choices=choices, usage=usage_info))
        return res

    @torch.inference_mode()
    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        # TODO:auto batch
        if request_config is None:
            request_config = RequestConfig()
        res_or_gen = self.infer([infer_request],
                                request_config,
                                template=template,
                                use_tqdm=False,
                                lora_request=lora_request)
        if request_config.stream:

            async def _gen_wrapper():
                for response in res_or_gen:
                    yield response[0]

            return _gen_wrapper()
        else:
            return res_or_gen[0]

    def _infer(
        self,
        infer_requests: List[InferRequest],
        request_config: RequestConfig,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        self.model.eval()
        request_config = deepcopy(request_config)
        if template is None:
            template = self.default_template
        if template.use_model:
            template.model = self.model

        template.set_mode('pt')
        batched_inputs = []
        template_inputs = []
        for infer_request in infer_requests:
            inputs = template.encode(infer_request, return_template_inputs=True)
            template_inputs.append(inputs.pop('template_inputs'))
            batched_inputs.append(inputs)
        inputs = to_device(template.data_collator(batched_inputs), self.model.device)
        if self.model.model_meta.is_multimodal:
            _, inputs = template.pre_forward_hook(self.model, None, inputs)
        self.set_default_max_tokens(request_config, inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template)

        kwargs = {
            'template': template,
            'inputs': inputs,
            'generation_config': generation_config,
            'lora_request': lora_request,
            'template_inputs': template_inputs
        }
        for pre_infer_hook in self.pre_infer_hooks:
            kwargs = pre_infer_hook(kwargs)
        if request_config.stream:

            def _gen_wrapper():
                for res in self._infer_stream(**kwargs):
                    yield res
                self._update_metrics(res, metrics)

            return _gen_wrapper()
        else:
            return self._update_metrics(self._infer_full(**kwargs), metrics)

    @torch.inference_mode()
    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        lora_request: Optional[LoRARequest] = None
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        if request_config is None:
            request_config = RequestConfig()
        if use_tqdm is None:
            use_tqdm = not request_config.stream and len(infer_requests) > 1
        prog_bar = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)

        if request_config.stream:

            def _gen_wrapper() -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:
                i = 0
                while i < len(infer_requests):
                    infer_requests_samples = infer_requests[i:i + self.max_batch_size]
                    gen = self._infer(
                        infer_requests_samples, request_config, metrics, template=template, lora_request=lora_request)
                    for response in gen:
                        res = [None] * len(infer_requests)
                        res[i:i + self.max_batch_size] = response
                        yield res
                    i += self.max_batch_size
                    prog_bar.update(len(infer_requests_samples))

            return _gen_wrapper()
        else:
            res = []
            i = 0
            while i < len(infer_requests):
                infer_requests_samples = infer_requests[i:i + self.max_batch_size]
                res += self._infer(
                    infer_requests_samples, request_config, metrics, template=template, lora_request=lora_request)
                i += self.max_batch_size
                prog_bar.update(len(infer_requests_samples))
            return res
