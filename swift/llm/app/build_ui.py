# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Literal, Optional

import gradio as gr

from swift.utils import get_file_mm_type
from ..utils import History
from .locale import locale_mapping


def clear_session():
    return '', [], []


def modify_system_session(system: str):
    system = system or ''
    return system, '', [], []


def _history_to_messages(history: History, system: Optional[str]):
    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    content = []
    for h in history:
        assert isinstance(h, (list, tuple))
        if isinstance(h[0], tuple):
            assert h[1] is None
            file_path = h[0][0]
            try:
                mm_type = get_file_mm_type(file_path)
                content.append({'type': mm_type, mm_type: file_path})
            except ValueError:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content.append({'type': 'text', 'text': f.read()})
        else:
            content.append({'type': 'text', 'text': h[0]})
            messages.append({'role': 'user', 'content': content})
            if h[1] is not None:
                messages.append({'role': 'assistant', 'content': h[1]})
            content = []
    return messages


def _parse_text(text: str) -> str:
    mapping = {'<': '&lt;', '>': '&gt;', '*': '&ast;'}
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


async def model_chat(history: History, real_history: History, system: Optional[str], *, client, model: str,
                     request_config: Optional['RequestConfig']):
    if history:
        from swift.llm import InferRequest

        messages = _history_to_messages(real_history, system)
        resp_or_gen = await client.infer_async(
            InferRequest(messages=messages), request_config=request_config, model=model)
        if request_config and request_config.stream:
            response = ''
            async for resp in resp_or_gen:
                if resp is None:
                    continue
                response += resp.choices[0].delta.content
                history[-1][1] = _parse_text(response)
                real_history[-1][-1] = response
                yield history, real_history

        else:
            response = resp_or_gen.choices[0].message.content
            history[-1][1] = _parse_text(response)
            real_history[-1][-1] = response
            yield history, real_history

    else:
        yield [], []


def add_text(history: History, real_history: History, query: str):
    history = history or []
    real_history = real_history or []
    history.append([_parse_text(query), None])
    real_history.append([query, None])
    return history, real_history, ''


def add_file(history: History, real_history: History, file):
    history = history or []
    real_history = real_history or []
    history.append([(file.name, ), None])
    real_history.append([(file.name, ), None])
    return history, real_history


def build_ui(base_url: str,
             model: Optional[str] = None,
             *,
             request_config: Optional['RequestConfig'] = None,
             is_multimodal: bool = True,
             studio_title: Optional[str] = None,
             lang: Literal['en', 'zh'] = 'en',
             default_system: Optional[str] = None):
    from swift.llm import InferClient
    client = InferClient(base_url=base_url)
    model = model or client.models[0]
    studio_title = studio_title or model
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>{studio_title}</center>')
        with gr.Row():
            with gr.Column(scale=3):
                system_input = gr.Textbox(value=default_system, lines=1, label='System')
            with gr.Column(scale=1):
                modify_system = gr.Button(locale_mapping['modify_system'][lang], scale=2)
        chatbot = gr.Chatbot(label='Chatbot')
        textbox = gr.Textbox(lines=1, label='Input')

        with gr.Row():
            upload = gr.UploadButton(locale_mapping['upload'][lang], visible=is_multimodal)
            submit = gr.Button(locale_mapping['submit'][lang])
            regenerate = gr.Button(locale_mapping['regenerate'][lang])
            clear_history = gr.Button(locale_mapping['clear_history'][lang])

        system_state = gr.State(value=default_system)
        history_state = gr.State(value=[])
        model_chat_ = partial(model_chat, client=client, model=model, request_config=request_config)

        upload.upload(add_file, [chatbot, history_state, upload], [chatbot, history_state])
        textbox.submit(add_text, [chatbot, history_state, textbox],
                       [chatbot, history_state, textbox]).then(model_chat_, [chatbot, history_state, system_state],
                                                               [chatbot, history_state])
        submit.click(add_text, [chatbot, history_state, textbox],
                     [chatbot, history_state, textbox]).then(model_chat_, [chatbot, history_state, system_state],
                                                             [chatbot, history_state])
        regenerate.click(model_chat_, [chatbot, history_state, system_state], [chatbot, history_state])
        clear_history.click(clear_session, [], [textbox, chatbot, history_state])
        modify_system.click(modify_system_session, [system_input], [system_state, textbox, chatbot, history_state])
    return demo
