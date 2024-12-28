import os
from http import HTTPStatus
from typing import Dict, List, Optional, Tuple

import gradio as gr

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


def clear_session():
    return '', []


def modify_system_session(system: str, default_system: str):
    system = system or default_system
    return system, system, []


def model_chat(query: str, history: History, system: str) -> Tuple[str, str, History]:
    query = query or ''
    history = history or []
    history.append([query, None])
    messages = history_to_messages(history, system)
    gen = Generation.call(model='qwen2-72b-instruct', messages=messages, result_format='message', stream=True)
    for response in gen:
        if response.status_code == HTTPStatus.OK:
            role = response.output.choices[0].message.role
            response = response.output.choices[0].message.content
            system, history = messages_to_history(messages + [{'role': role, 'content': response}])
            yield '', history, system
        else:
            raise ValueError('Request id: %s, Status code: %s, error code: %s, error message: %s' %
                             (response.request_id, response.status_code, response.code, response.message))


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


def build_llm_ui(studio_title: str, *, lang: str = 'zh', default_system: Optional[str] = None):
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
        textbox.submit(model_chat, inputs=[textbox, chatbot, system_state], outputs=[textbox, chatbot, system_input])

        submit.click(
            model_chat,
            inputs=[textbox, chatbot, system_state],
            outputs=[textbox, chatbot, system_input],
            concurrency_limit=5)
        clear_history.click(fn=clear_session, inputs=[], outputs=[textbox, chatbot])
        modify_system.click(
            fn=modify_system_session,
            inputs=[system_input, default_system],
            outputs=[system_state, system_input, chatbot])
    return demo
