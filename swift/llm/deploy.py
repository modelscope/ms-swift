# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from dataclasses import asdict
from http import HTTPStatus
from typing import List, Optional, Union

import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from swift.utils import get_main, seed_everything
from .infer import merge_lora, prepare_model_template
from .utils import ChatCompletionResponse  # noqa
from .utils import (ChatCompletionRequest, ChatCompletionResponseChoice,
                    ChatCompletionResponseStreamChoice,
                    ChatCompletionStreamResponse, ChatMessage,
                    CompletionRequest, CompletionResponse,
                    CompletionResponseChoice, CompletionResponseStreamChoice,
                    CompletionStreamResponse, DeltaMessage, DeployArguments,
                    Model, ModelList, UsageInfo, VllmGenerationConfig,
                    messages_to_history, prepare_vllm_engine_template,
                    random_uuid)

app = FastAPI()
model = None
llm_engine = None
template = None


def create_error_response(status_code: Union[int, str, HTTPStatus],
                          message: str) -> JSONResponse:
    status_code = int(status_code)
    return JSONResponse({'message': message, 'object': 'error'}, status_code)


@app.get('/v1/models')
async def get_available_models():
    global llm_engine
    return ModelList(data=[Model(id=llm_engine.model_type)])


async def check_length(request: Union[ChatCompletionRequest,
                                      CompletionRequest],
                       input_ids: List[int]) -> Optional[str]:
    global llm_engine
    max_model_len = llm_engine.model_config.max_model_len
    num_tokens = len(input_ids)
    max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = max_model_len - num_tokens
        request.max_tokens = max_tokens
    if max_tokens + num_tokens > max_model_len:
        error_msg = (
            f'Your prompt has {num_tokens} tokens, and you have set the `max_tokens` to {max_tokens}, '
            f'but the maximum model length supported is {max_model_len}. '
            'Please reduce the number of tokens in the prompt or the `max_tokens`.'
        )
        return error_msg


async def check_model(
        request: Union[ChatCompletionRequest,
                       CompletionRequest]) -> Optional[str]:
    model_list = await get_available_models()
    model_type_list = [model.id for model in model_list.data]
    if request.model in model_type_list:
        return
    else:
        return f'`{request.model}` is not in the model_list: `{model_type_list}`.'


async def inference_vllm_async(request: Union[ChatCompletionRequest,
                                              CompletionRequest],
                               raw_request: Request):
    global llm_engine, template
    error_msg = await check_model(request)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

    if request.seed is not None:
        seed_everything(request.seed)
    if isinstance(request, ChatCompletionRequest):
        if is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                f'The chat template `{template.template_type}` corresponding to '
                f'the model `{llm_engine.model_type}` is in text generation format. '
                'Please use the `completions` API.')
        example = messages_to_history(request.messages)
        input_ids = template.encode(example)[0]['input_ids']
        request_id = f'chatcmpl-{random_uuid()}'
    else:
        if not is_generation_template(template.template_type):
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                f'The chat template `{template.template_type}` corresponding to '
                f'the model `{llm_engine.model_type}` is in chat format. '
                'Please use the `chat.completions` API.')
        input_ids = template.encode({'query': request.prompt})[0]['input_ids']
        request_id = f'cmpl-{random_uuid()}'

    error_msg = await check_length(request, input_ids)
    if error_msg is not None:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_msg)
    kwargs = {'max_new_tokens': request.max_tokens}
    for key in [
            'n', 'stop', 'best_of', 'frequency_penalty', 'presence_penalty',
            'num_beams'
    ]:
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
    created_time = int(time.time())
    result_generator = llm_engine.generate(None, generation_config, request_id,
                                           input_ids)

    async def _generate_full():
        result = None
        async for result in result_generator:
            if await raw_request.is_disconnected():
                await llm_engine.abort(request_id)
                return create_error_response(HTTPStatus.BAD_REQUEST,
                                             'Client disconnected')
        assert result is not None
        num_prompt_tokens = len(result.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in result.outputs)
        usage_info = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if isinstance(request, ChatCompletionRequest):
            choices = []
            for output in result.outputs:
                choice = ChatCompletionResponseChoice(
                    index=output.index,
                    message=ChatMessage(role='assistant', content=output.text),
                    finish_reason=output.finish_reason,
                )
                choices.append(choice)
            return ChatCompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage_info,
                id=request_id,
                created=created_time)
        else:
            choices = []
            for output in result.outputs:
                choice = CompletionResponseChoice(
                    index=output.index,
                    text=output.text,
                    finish_reason=output.finish_reason,
                )
                choices.append(choice)
            return CompletionResponse(
                model=request.model,
                choices=choices,
                usage=usage_info,
                id=request_id,
                created=created_time)

    async def _generate_stream():
        print_idx_list = [0] * request.n
        async for result in result_generator:
            num_prompt_tokens = len(result.prompt_token_ids)
            num_generated_tokens = sum(
                len(output.token_ids) for output in result.outputs)
            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            if isinstance(request, ChatCompletionRequest):
                choices = []
                for output in result.outputs:
                    delta_text = output.text[print_idx_list[output.index]:]
                    print_idx_list[output.index] += len(delta_text)
                    choice = ChatCompletionResponseStreamChoice(
                        index=output.index,
                        delta=DeltaMessage(
                            role='assistant', content=delta_text),
                        finish_reason=output.finish_reason)
                    choices.append(choice)
                response = ChatCompletionStreamResponse(
                    model=request.model,
                    choices=choices,
                    usage=usage_info,
                    id=request_id,
                    created=created_time)
            else:
                choices = []
                for output in result.outputs:
                    delta_text = output.text[print_idx_list[output.index]:]
                    print_idx_list[output.index] += len(delta_text)
                    choice = CompletionResponseStreamChoice(
                        index=output.index,
                        text=delta_text,
                        finish_reason=output.finish_reason)
                    choices.append(choice)
                response = CompletionStreamResponse(
                    model=request.model,
                    choices=choices,
                    usage=usage_info,
                    id=request_id,
                    created=created_time)
            yield f'data:{json.dumps(asdict(response), ensure_ascii=False)}\n\n'
        yield 'data:[DONE]\n\n'

    if request.stream:
        return StreamingResponse(_generate_stream())
    else:
        return await _generate_full()


def is_generation_template(template_type: str) -> bool:
    if 'generation' in template_type:
        return True
    else:
        return False


@app.post('/v1/chat/completions')
async def create_chat_completion(
        request: ChatCompletionRequest,
        raw_request: Request) -> ChatCompletionResponse:
    return await inference_vllm_async(request, raw_request)


@app.post('/v1/completions')
async def create_completion(request: CompletionRequest,
                            raw_request: Request) -> CompletionResponse:
    return await inference_vllm_async(request, raw_request)


def llm_deploy(args: DeployArguments) -> None:
    import uvicorn
    global llm_engine, model, template
    if args.merge_lora_and_save:
        merge_lora(args, device_map='cpu')
    if args.infer_backend == 'vllm':
        llm_engine, template = prepare_vllm_engine_template(
            args, use_async=True)
    else:
        model, template = prepare_model_template(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile)


deploy_main = get_main(DeployArguments, llm_deploy)
