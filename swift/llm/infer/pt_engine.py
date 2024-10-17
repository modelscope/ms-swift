import inspect
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import json
import torch
from transformers import GenerationConfig, PreTrainedTokenizerBase, StoppingCriteriaList
from transformers.utils import is_torch_npu_available

from swift.plugin import Metric
from swift.utils import get_logger
from ..template import Template
from ..utils import to_device
from .base import InferEngine
from .protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                       ChatCompletionStreamResponse, ChatMessage, DeltaMessage, InferRequest, RequestConfig)
from .utils import InferStreamer, InferTools, StopWordsCriteria, TokensIteratorStreamer

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

    def __init__(self,
                 model_id_or_path: str,
                 torch_dtype: Optional[torch.dtype] = None,
                 *,
                 model_type: Optional[str] = None,
                 **kwargs):
        self._prepare_model_tokenizer(model_id_or_path, torch_dtype, True, model_type=model_type, **kwargs)
        self.engine = self.model
        self.generation_config = self.model.generation_config

    def _prepare_generation_config(self, request_config: RequestConfig) -> GenerationConfig:
        kwargs = {'max_new_tokens': request_config.max_tokens}
        # not use: 'n', 'best_of', 'frequency_penalty', 'presence_penalty'
        for key in ['length_penalty']:
            kwargs[key] = getattr(request_config, key)
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty', 'num_beams']:
            new_value = getattr(request_config, key)
            if new_value is None:
                kwargs[key] = getattr(self.generation_config, key)
            else:
                kwargs[key] = new_value

        if not self.generation_config.do_sample:
            kwargs['temperature'] = 0
        if kwargs['temperature'] == 0:
            kwargs['do_sample'] = False
            kwargs['temperature'] = 1
            kwargs['top_p'] = 1
            kwargs['top_k'] = 50
        else:
            kwargs['do_sample'] = True
        kwargs['return_dict_in_generate'] = True
        if request_config.logprobs:
            kwargs['output_logits'] = True
        generation_config = _GenerationConfig(**kwargs)
        generation_config.top_logprobs = request_config.top_logprobs
        return self._set_generation_config_default_value(generation_config)

    def _set_generation_config_default_value(self, generation_config: GenerationConfig) -> GenerationConfig:
        for k, v in self.generation_config.to_dict().items():
            new_v = getattr(generation_config, k, None)
            if k in ['max_length']:
                continue
            if k in ['no_repeat_ngram_size'] or v is not None and new_v is None:
                setattr(generation_config, k, v)
        return generation_config

    def _add_stop_words(self, generation_config: GenerationConfig, request_config: RequestConfig,
                        template: Template) -> None:
        stop_words = (request_config.stop or []) + (self.generation_config.stop or []) + template.stop_words
        stop_words += [template.suffix[-1], self.tokenizer.eos_token]
        generation_config.stop_words = self._get_stop_words(stop_words)

    @staticmethod
    def _get_logprobs(tokenizer: PreTrainedTokenizerBase,
                      logits_list: Optional[List[torch.Tensor]],
                      generate_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if logits_list is None:
            return None
        res = []
        for logits, token_id in zip(logits_list, generate_ids):
            token = tokenizer.decode(token_id)
            logprobs = torch.log_softmax(logits[0], -1)
            logprob = logprobs[token_id].item()
            sorted_logprobs_idx = logprobs.argsort(descending=True).tolist()
            _res = {'token': token, 'logprob': logprob, 'bytes': list(token.encode('utf8'))}
            if top_logprobs is not None:
                res_top_logprobs = []
                for idx in sorted_logprobs_idx[:top_logprobs]:
                    token = tokenizer.decode(idx)
                    logprob = logprobs[idx].item()
                    if idx == token_id or logprob == float('-inf'):
                        continue
                    res_top_logprobs.append({'token': token, 'logprob': logprob, 'bytes': list(token.encode('utf8'))})
                _res['top_logprobs'] = res_top_logprobs
            res.append(_res)
        return {'content': res}

    async def _infer_stream_async(
            self,
            template: Template,
            infer_request: InferRequest,
            request_config: RequestConfig,
            *,
            adapter_names: Optional[List[str]] = None) -> AsyncIterator[ChatCompletionStreamResponse]:
        gen = self.infer(template, [infer_request], request_config, use_tqdm=False, adapter_names=adapter_names)
        for response in gen:
            yield response[0]

    async def _infer_full_async(self,
                                template: Template,
                                infer_request: InferRequest,
                                request_config: RequestConfig,
                                *,
                                adapter_names: Optional[List[str]] = None) -> ChatCompletionResponse:
        return self.infer(template, [infer_request], request_config, use_tqdm=False, adapter_names=adapter_names)[0]

    def __model_generate(self, *args, **kwargs):
        queue = kwargs.pop('queue')
        if is_torch_npu_available():
            torch.npu.set_device(self.model.device)
        res = self.model.generate(*args, **kwargs)
        queue.put(res)

    @staticmethod
    def _get_finish_reason(generation_config: GenerationConfig, num_prompt_tokens: int, is_finished: bool):
        if is_finished:
            if num_prompt_tokens >= generation_config.max_new_tokens:
                finish_reason = 'length'
            else:
                finish_reason = 'stop'
        else:
            finish_reason = None
        return finish_reason

    def _infer_stream(self,
                      template: Template,
                      inputs: Dict[str, Any],
                      generation_config: GenerationConfig,
                      *,
                      adapter_names: Optional[List[str]] = None) -> Iterator[List[ChatCompletionStreamResponse]]:
        if adapter_names is not None:
            inputs['adapter_names'] = adapter_names
        stopping_criteria = StoppingCriteriaList([StopWordsCriteria(self.tokenizer, generation_config.stop_words)])
        if generation_config.num_beams != 1:
            error_msg = 'Streaming generation does not support beam search.'
            raise ValueError(error_msg)

        streamer = TokensIteratorStreamer()
        thread = Thread(
            target=self.__model_generate,
            kwargs={
                'generation_config': generation_config,
                'stopping_criteria': stopping_criteria,
                'streamer': streamer,
                **inputs
            })
        thread.start()
        num_prompt_tokens = InferEngine._get_num_tokens(inputs)
        infer_stream = InferStreamer(template)
        raw_generate_ids = []
        total_response = ''
        is_finished = False
        while not is_finished:
            try:
                raw_generate_ids += next(streamer)
            except StopIteration:
                is_finished = True
            generate_ids = template.get_generate_ids(raw_generate_ids, num_prompt_tokens)
            delta_text = infer_stream.get_printable_text(generate_ids, is_finished)
            if not delta_text and not is_finished:
                continue
            num_generated_tokens = len(generate_ids)
            usage_info = InferEngine._get_usage_info(num_prompt_tokens, num_generated_tokens)
            total_response += delta_text
            finish_reason = PtEngine._get_finish_reason(generation_config, num_prompt_tokens, is_finished)
            toolcall = InferEngine._get_toolcall(total_response, is_finished)
            choices = [
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),
                    finish_reason=finish_reason)
            ]
            yield ChatCompletionStreamResponse(model=self.model_type, choices=choices, usage=usage_info)

    def _infer_full(self,
                    template: Template,
                    inputs: Dict[str, Any],
                    generation_config: GenerationConfig,
                    *,
                    adapter_names: Optional[List[str]] = None) -> List[ChatCompletionResponse]:
        if adapter_names is not None:
            inputs['adapter_names'] = adapter_names
        stopping_criteria = StoppingCriteriaList([StopWordsCriteria(self.tokenizer, generation_config.stop_words)])
        res = dict(
            self.model.generate(generation_config=generation_config, stopping_criteria=stopping_criteria, **inputs))
        generate_ids = res['sequences']
        num_prompt_tokens = InferEngine._get_num_tokens(inputs)
        num_generated_tokens = len(generate_ids)
        generate_ids = template.get_generate_ids(generate_ids, num_prompt_tokens)
        usage_info = InferEngine._get_usage_info(num_prompt_tokens, num_generated_tokens)
        response = InferTools.safe_decode(template, generate_ids, True)
        logprobs = self._get_logprobs(self.tokenizer, res['logits'], generate_ids, generation_config.top_logprobs)
        toolcall = InferEngine._get_toolcall(response, True)
        choices = [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=None,
                logprobs=logprobs)
        ]
        return [ChatCompletionResponse(model=self.model_type, choices=choices, usage=usage_info)]

    @torch.inference_mode()
    async def infer_async(
        self,
        template: Template,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        adapter_names: Optional[List[str]] = None
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        infer_args = (template, infer_request, request_config)
        if request_config.stream:
            return self._infer_stream_async(*infer_args, adapter_names=adapter_names)
        else:
            return await self._infer_full_async(*infer_args, adapter_names=adapter_names)

    @torch.inference_mode()
    def infer(
        self,
        template: Template,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
        metrics: Optional[List[Metric]] = None,
        adapter_names: Optional[List[str]] = None
    ) -> Union[List[ChatCompletionResponse], Iterator[List[ChatCompletionStreamResponse]]]:
        self.model.eval()
        request_config = request_config or RequestConfig()

        batched_inputs = []
        for infer_request in infer_requests:
            inputs = template.encode(infer_request)
            assert len(inputs) >= 0
            batched_inputs.append(inputs)
        self.set_default_max_tokens(request_config, batched_inputs)
        generation_config = self._prepare_generation_config(request_config)
        self._add_stop_words(generation_config, request_config, template)

        inputs = to_device(template.data_collator(batched_inputs), next(self.model.parameters()).device)
        infer_args = (template, inputs, generation_config)
        if request_config.stream:
            return self._infer_stream(*infer_args, adapter_names=adapter_names)
        else:
            return self._infer_full(*infer_args, adapter_names=adapter_names)
