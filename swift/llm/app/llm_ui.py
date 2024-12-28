import os
from functools import partial
from http import HTTPStatus
from typing import Dict, List, Literal, Optional, Tuple

import gradio as gr

from ..utils import history_to_messages

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


def clear_session():
    return '', []


def modify_system_session(system: str):
    system = system or ''
    return system, system, []


def model_chat(query: str, history: History, system: str, *, base_url: str) -> Tuple[str, str, History]:
    from swift.llm import InferRequest, InferClient, RequestConfig
    query = query or ''
    history = history or []
    history.append([query, None])
    messages = history_to_messages(history, system)
    client = InferClient(base_url=base_url)
    gen = client.infer([InferRequest(messages=messages)], request_config=RequestConfig(stream=True))
    response = ''
    for resp_list in gen:
        resp = resp_list[0]
        if resp is None:
            continue
        response += resp.choices[0].message.delta
        history[-1][1] = response
        yield '', history, system


locale_mapping = {
    'modify_system': {
        'en': '🛠️ Set system and clear history',
        'zh': '🛠️ 设置system并清空历史'
    },
    'clear_history': {
        'en': '🧹 Clear history',
        'zh': '🧹 清空历史'
    },
    'submit': {
        'en': '🚀 Send',
        'zh': '🚀 发送'
    },
}


def build_llm_ui(base_url: str,
                 *,
                 studio_title: Optional[str] = None,
                 lang: Literal['en', 'zh'] = 'zh',
                 default_system: Optional[str] = None):
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
            clear_history = gr.Button(locale_mapping['clear_history'][lang])
            submit = gr.Button(locale_mapping['submit'][lang])

        system_state = gr.State(value=default_system)
        model_chat_ = partial(model_chat, base_url=base_url)
        textbox.submit(model_chat_, inputs=[textbox, chatbot, system_state], outputs=[textbox, chatbot, system_input])

        submit.click(
            model_chat_,
            inputs=[textbox, chatbot, system_state],
            outputs=[textbox, chatbot, system_input],
            concurrency_limit=5)
        clear_history.click(fn=clear_session, inputs=[], outputs=[textbox, chatbot])
        modify_system.click(
            fn=modify_system_session, inputs=[system_input], outputs=[system_state, system_input, chatbot])
    return demo
