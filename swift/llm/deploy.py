# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import logging
import time
from dataclasses import asdict
from http import HTTPStatus
from typing import List, Optional, Union

import json
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from modelscope import GenerationConfig
from peft import PeftModel

from swift.utils import get_logger, get_main, seed_everything
from .infer import merge_lora, prepare_model_template
from .utils import ChatCompletionResponse  # noqa
from .utils import (ChatCompletionRequest, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                    ChatCompletionStreamResponse, ChatMessage, CompletionRequest, CompletionResponse,
                    CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse, DeltaMessage,
                    DeployArguments, Model, ModelList, UsageInfo, inference, inference_stream, messages_to_history,
                    random_uuid)

logger = get_logger()

app = FastAPI()
_args = None
model = None
llm_engine = None
template = None


def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
    status_code = int(status_code)
    return JSONResponse({'message': message, 'object': 'error'}, status_code)


@app.get('/v1/models')
async def get_available_models():
    global _args
    model_list = [_args.model_type]
    if _args.lora_request_list is not None:
        model_list += [lora_request.lora_name for lora_request in _args.lora_request_list]
    data = [Model(id=model_id) for model_id in model_list]
    return ModelList(data=data)


async def check_length(request: Union[ChatCompletionRequest, CompletionRequest], input_ids: List[int]) -> Optional[str]:
    global llm_engine, model, _args
    if _args.infer_backend == 'vllm':
        max_model_len = llm_engine.model_config.max_model_len
    else:
        max_model_len = model.max_model_len
    num_tokens = len(input_ids)
    max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = max_model_len - num_tokens
        request.max_tokens = max_tokens
    if max_tokens + num_tokens > max_model_len:
        error_msg = (f'Your prompt has {num_tokens} tokens, and you have set the `max_tokens` to {max_tokens}, '
                     f'but the maximum model length supported is {max_model_len}. '
                     'Please reduce the number of tokens in the prompt or the `max_tokens`.')
        return error_msg


async def check_model(request: Union[ChatCompletionRequest, CompletionRequest]) -> Optional[str]:
    model_list = await get_available_models()
    model_type_list = [model.id for model in model_list.data]
    if request.model in model_type_list:
        return
    else:
        return f'`{request.model}` is not in the model_list: `{model_type_list}`.'


def is_generation_template(template_type: str) -> bool:
    if 'generation' in template_type:
        return True
    else:
        return False


@torch.inference_mode()
async def inference_vllm_async(request: Union[ChatCompletionRequest, CompletionRequest], raw_request: Request):
    global llm_engine, template, _args
    from .utils import VllmGenerationConfig
    error_msg = await check_model(request)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

    if request.seed is not None:
        seed_everything(request.seed, verbose=False)
    _request = {'model': request.model}
    if isinstance(request, ChatCompletionRequest):
        if is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f'The chat template `{template.template_type}` corresponding to '
                f'the model `{llm_engine.model_type}` is in text generation format. '
                'Please use the `completions` API.')
        example = messages_to_history(request.messages)
        input_ids = template.encode(example)[0]['input_ids']
        request_id = f'chatcmpl-{random_uuid()}'
        _request['messages'] = request.messages
    else:
        if not is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f'The chat template `{template.template_type}` corresponding to '
                f'the model `{llm_engine.model_type}` is in chat format. '
                'Please use the `chat.completions` API.')
        example = {'query': request.prompt}
        input_ids = template.encode(example)[0]['input_ids']
        request_id = f'cmpl-{random_uuid()}'
        _request['prompt'] = request.prompt

    request_info = {'request_id': request_id}
    request_info.update(_request)

    error_msg = await check_length(request, input_ids)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

    kwargs = {'max_new_tokens': request.max_tokens}
    for key in ['n', 'stop', 'best_of', 'frequency_penalty', 'length_penalty', 'presence_penalty', 'num_beams']:
        kwargs[key] = getattr(request, key)
    for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
        new_value = getattr(request, key)
        if new_value is None:
            kwargs[key] = getattr(llm_engine.generation_config, key)
        else:
            kwargs[key] = new_value

    generation_config = VllmGenerationConfig(**kwargs)
    if generation_config.use_beam_search is True and request.stream is True:
        error_msg = 'Streaming generation does not support beam search.'
        raise ValueError(error_msg)
    tokenizer = template.tokenizer
    if tokenizer.eos_token is not None and tokenizer.eos_token not in generation_config.stop:
        generation_config.stop.append(tokenizer.eos_token)
    if isinstance(template.suffix[-1], str) and template.suffix[-1] not in generation_config.stop:
        generation_config.stop.append(template.suffix[-1])
    if isinstance(template.suffix[-1], list):
        token_str = tokenizer.decode(template.suffix[-1])
        if token_str not in generation_config.stop:
            generation_config.stop.append(token_str)
    request_info['generation_config'] = generation_config
    request_info.update({'seed': request.seed, 'stream': request.stream})
    logger.info(request_info)

    created_time = int(time.time())
    generate_kwargs = {}
    if _args.vllm_enable_lora and request.model != _args.model_type:
        lora_request = None
        for lora_req in _args.lora_request_list:
            if lora_req.lora_name == request.model:
                lora_request = lora_req
                break
        assert lora_request is not None
        generate_kwargs['lora_request'] = lora_request
    result_generator = llm_engine.generate(None, generation_config, request_id, input_ids, **generate_kwargs)

    async def _generate_full():
        result = None
        async for result in result_generator:
            if await raw_request.is_disconnected():
                await llm_engine.abort(request_id)
                return create_error_response(HTTPStatus.BAD_REQUEST, 'Client disconnected')
        assert result is not None
        num_prompt_tokens = len(result.prompt_token_ids)
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        if isinstance(request, ChatCompletionRequest):
            choices = []
            for output in result.outputs:
                response = template.generate_ids_to_response(output.token_ids)
                choice = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role='assistant', content=response),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice)
            response = ChatCompletionResponse(
                model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
        else:
            choices = []
            for output in result.outputs:
                response = template.generate_ids_to_response(output.token_ids)
                choice = CompletionResponseChoice(
                    index=output.index,
                    text=response,
                    finish_reason=output.finish_reason,
                )
                choices.append(choice)
            response = CompletionResponse(
                model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
        return response

    async def _generate_stream():
        print_idx_list = [[0] for _ in range(request.n)]
        async for result in result_generator:
            num_prompt_tokens = len(result.prompt_token_ids)
            num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )

            for output in result.outputs:
                output.delta_text = template.generate_ids_to_response(
                    output.token_ids, output.finished(), return_delta=True, print_idx=print_idx_list[output.index])

            if isinstance(request, ChatCompletionRequest):
                choices = []
                for output in result.outputs:
                    choice = ChatCompletionResponseStreamChoice(
                        index=output.index,
                        delta=DeltaMessage(role='assistant', content=output.delta_text),
                        finish_reason=output.finish_reason)
                    choices.append(choice)
                response = ChatCompletionStreamResponse(
                    model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
            else:
                choices = []
                for output in result.outputs:
                    choice = CompletionResponseStreamChoice(
                        index=output.index, text=output.delta_text, finish_reason=output.finish_reason)
                    choices.append(choice)
                response = CompletionStreamResponse(
                    model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
            yield f'data:{json.dumps(asdict(response), ensure_ascii=False)}\n\n'
        yield 'data:[DONE]\n\n'

    if request.stream:
        return StreamingResponse(_generate_stream())
    else:
        return await _generate_full()


class _GenerationConfig(GenerationConfig):

    def __repr__(self) -> str:
        parameters = inspect.signature(self.to_json_string).parameters
        kwargs = {}
        if 'ignore_metadata' in parameters:
            kwargs['ignore_metadata'] = True
        gen_kwargs = json.loads(self.to_json_string(**kwargs))
        gen_kwargs.pop('transformers_version', None)
        return f'GenerationConfig({gen_kwargs})'


@torch.inference_mode()
async def inference_pt_async(request: Union[ChatCompletionRequest, CompletionRequest], raw_request: Request):
    global model, template
    error_msg = await check_model(request)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

    if request.seed is not None:
        seed_everything(request.seed, verbose=False)
    _request = {'model': request.model}
    if isinstance(request, ChatCompletionRequest):
        if is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f'The chat template `{template.template_type}` corresponding to '
                f'the model `{model.model_type}` is in text generation format. '
                'Please use the `completions` API.')
        example = messages_to_history(request.messages)
        input_ids = template.encode(example)[0]['input_ids']
        request_id = f'chatcmpl-{random_uuid()}'
        _request['messages'] = request.messages
    else:
        if not is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST, f'The chat template `{template.template_type}` corresponding to '
                f'the model `{model.model_type}` is in chat format. '
                'Please use the `chat.completions` API.')
        example = {'query': request.prompt}
        input_ids = template.encode(example)[0]['input_ids']
        request_id = f'cmpl-{random_uuid()}'
        _request['prompt'] = request.prompt

    request_info = {'request_id': request_id}
    request_info.update(_request)

    error_msg = await check_length(request, input_ids)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

    kwargs = {'max_new_tokens': request.max_tokens}
    # not use: 'n', 'best_of', 'frequency_penalty', 'presence_penalty'
    for key in ['length_penalty', 'num_beams']:
        kwargs[key] = getattr(request, key)
    for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:
        new_value = getattr(request, key)
        if new_value is None:
            kwargs[key] = getattr(model.generation_config, key)
        else:
            kwargs[key] = new_value
    if kwargs['temperature'] == 0:
        kwargs['do_sample'] = False
        kwargs['temperature'] = 1
        kwargs['top_p'] = 1
        kwargs['top_k'] = 50
    else:
        kwargs['do_sample'] = True

    generation_config = _GenerationConfig(**kwargs)
    request_info['generation_config'] = generation_config
    request_info.update({'seed': request.seed, 'stop': request.stop, 'stream': request.stream})
    logger.info(request_info)

    created_time = int(time.time())
    adapter_kwargs = {}
    if request.model != _args.model_type:
        adapter_names = None
        for lora_req in _args.lora_request_list:
            if lora_req.lora_name == request.model:
                adapter_names = request.model
                break
        assert adapter_names is not None
        adapter_kwargs['adapter_names'] = [adapter_names]
    elif isinstance(model, PeftModel):
        adapter_kwargs['adapter_names'] = ['-']

    async def _generate_full():
        generation_info = {}
        response, _ = inference(
            model,
            template,
            **example,
            stop_words=request.stop,
            generation_config=generation_config,
            generation_info=generation_info,
            **adapter_kwargs)
        num_prompt_tokens = generation_info['num_prompt_tokens']
        num_generated_tokens = generation_info['num_generated_tokens']
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if isinstance(request, ChatCompletionRequest):
            choices = [
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role='assistant', content=response),
                    finish_reason=None,
                )
            ]
            response = ChatCompletionResponse(
                model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
        else:
            choices = [CompletionResponseChoice(
                index=0,
                text=response,
                finish_reason=None,
            )]
            response = CompletionResponse(
                model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
        return response

    def _generate_stream():
        generation_info = {}
        gen = inference_stream(
            model,
            template,
            **example,
            stop_words=request.stop,
            generation_config=generation_config,
            generation_info=generation_info,
            **adapter_kwargs)

        print_idx = 0
        for response, _ in gen:
            num_prompt_tokens = generation_info['num_prompt_tokens']
            num_generated_tokens = generation_info['num_generated_tokens']
            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            if isinstance(request, ChatCompletionRequest):
                delta_text = response[print_idx:]
                print_idx = len(response)
                choices = [
                    ChatCompletionResponseStreamChoice(
                        index=0, delta=DeltaMessage(role='assistant', content=delta_text), finish_reason=None)
                ]
                resp = ChatCompletionStreamResponse(
                    model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
            else:
                delta_text = response[print_idx:]
                print_idx = len(response)
                choices = [CompletionResponseStreamChoice(index=0, text=delta_text, finish_reason=None)]
                resp = CompletionStreamResponse(
                    model=request.model, choices=choices, usage=usage_info, id=request_id, created=created_time)
            yield f'data:{json.dumps(asdict(resp), ensure_ascii=False)}\n\n'
        yield 'data:[DONE]\n\n'

    if request.stream:
        return StreamingResponse(_generate_stream())
    else:
        return await _generate_full()


@app.post('/v1/chat/completions')
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> ChatCompletionResponse:
    global _args
    assert _args is not None
    if _args.infer_backend == 'vllm':
        return await inference_vllm_async(request, raw_request)
    else:
        return await inference_pt_async(request, raw_request)


@app.post('/v1/completions')
async def create_completion(request: CompletionRequest, raw_request: Request) -> CompletionResponse:
    global _args
    assert _args is not None
    if _args.infer_backend == 'vllm':
        return await inference_vllm_async(request, raw_request)
    else:
        return await inference_pt_async(request, raw_request)


def llm_deploy(args: DeployArguments) -> None:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    logger_format = logging.Formatter('%(levelname)s: %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    logger.handlers[0].setFormatter(logger_format)
    import uvicorn
    global llm_engine, model, template, _args
    _args = args
    if args.merge_lora:
        merge_lora(args, device_map=args.merge_device_map)
    if args.infer_backend == 'vllm':
        from .utils import prepare_vllm_engine_template
        llm_engine, template = prepare_vllm_engine_template(args, use_async=True)
    else:
        model, template = prepare_model_template(args)
    uvicorn.run(app, host=args.host, port=args.port, ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile)


deploy_main = get_main(DeployArguments, llm_deploy)
