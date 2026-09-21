"""Gradio chat application over a remote or temporary dev deployment."""
from __future__ import annotations
from contextlib import nullcontext
from functools import partial
from typing import Optional

_LOCALE = {
    'modify_system': {'en': '🛠️ Set system and clear history', 'zh': '🛠️ 设置 system 并清空历史'},
    'clear_history': {'en': '🧹 Clear history', 'zh': '🧹 清空历史'},
    'submit': {'en': '🚀 Send', 'zh': '🚀 发送'},
    'regenerate': {'en': '🤔️ Regenerate', 'zh': '🤔️ 重试'},
    'upload': {'en': '📁 Upload', 'zh': '📁 上传'},
}


def _clear_session():
    return '', [], []


def _modify_system(system: str):
    return system or '', '', [], []


def _parse_text(text: str) -> str:
    for source, target in {'<': '&lt;', '>': '&gt;', '*': '&ast;'}.items():
        text = text.replace(source, target)
    return text


def _history_to_messages(history, system: Optional[str]):
    from swift.utils import get_file_mm_type

    messages = []
    if system is not None:
        messages.append({'role': 'system', 'content': system})
    content = []
    for item in history:
        if isinstance(item[0], tuple):
            path = item[0][0]
            try:
                media_type = get_file_mm_type(path)
                content.append({'type': media_type, media_type: path})
            except ValueError:
                with open(path, encoding='utf-8') as file:
                    content.append({'type': 'text', 'text': file.read()})
        else:
            content.append({'type': 'text', 'text': item[0]})
            messages.append({'role': 'user', 'content': content})
            if item[1] is not None:
                messages.append({'role': 'assistant', 'content': item[1]})
            content = []
    return messages


async def _model_chat(history, real_history, system, *, client, model, request_config):
    from swift.infer_engine import InferRequest

    if not history:
        yield [], []
        return
    response = await client.infer_async(
        InferRequest(messages=_history_to_messages(real_history, system)), request_config=request_config, model=model)
    if request_config.stream:
        text = ''
        async for chunk in response:
            if chunk is None:
                continue
            text += chunk.choices[0].delta.content
            history[-1][1] = _parse_text(text)
            real_history[-1][-1] = text
            yield history, real_history
    else:
        text = response.choices[0].message.content
        history[-1][1] = _parse_text(text)
        real_history[-1][-1] = text
        yield history, real_history


def _add_text(history, real_history, query: str):
    history, real_history = history or [], real_history or []
    history.append([_parse_text(query), None])
    real_history.append([query, None])
    return history, real_history, ''


def _add_file(history, real_history, file):
    history, real_history = history or [], real_history or []
    history.append([(file.name,), None])
    real_history.append([(file.name,), None])
    return history, real_history


def build_app_ui(base_url: str, model: Optional[str], request_config, app_config, default_system: Optional[str] = None):
    """Build the Gradio app without launching it."""
    import gradio as gr

    from swift.infer_engine import InferClient

    client = InferClient(base_url=base_url)
    model = model or client.models[0]
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>{app_config.studio_title or model}</center>')
        with gr.Row():
            with gr.Column(scale=3):
                system_input = gr.Textbox(value=default_system, lines=1, label='System')
            with gr.Column(scale=1):
                modify_system = gr.Button(_LOCALE['modify_system'][app_config.lang], scale=2)
        chatbot = gr.Chatbot(label='Chatbot')
        textbox = gr.Textbox(lines=1, label='Input')
        with gr.Row():
            upload = gr.UploadButton(_LOCALE['upload'][app_config.lang], visible=app_config.is_multimodal)
            submit = gr.Button(_LOCALE['submit'][app_config.lang])
            regenerate = gr.Button(_LOCALE['regenerate'][app_config.lang])
            clear_history = gr.Button(_LOCALE['clear_history'][app_config.lang])
        system_state = gr.State(value=default_system)
        history_state = gr.State(value=[])
        chat = partial(_model_chat, client=client, model=model, request_config=request_config)
        upload.upload(_add_file, [chatbot, history_state, upload], [chatbot, history_state])
        textbox.submit(_add_text, [chatbot, history_state, textbox], [chatbot, history_state, textbox]).then(
            chat, [chatbot, history_state, system_state], [chatbot, history_state])
        submit.click(_add_text, [chatbot, history_state, textbox], [chatbot, history_state, textbox]).then(
            chat, [chatbot, history_state, system_state], [chatbot, history_state])
        regenerate.click(chat, [chatbot, history_state, system_state], [chatbot, history_state])
        clear_history.click(_clear_session, [], [textbox, chatbot, history_state])
        modify_system.click(_modify_system, [system_input], [system_state, textbox, chatbot, history_state])
    return demo


def run_app(model_config, template_config, generation_config, infer_config, rollout_config, deploy_config, app_config,
            *, adapter_mapping=None, merge_lora: bool = False) -> None:
    """Launch the UI, starting a temporary dev deployment when base_url is absent."""
    import gradio
    from packaging import version

    from swift.dev.cli.infer import _engine_args
    from swift.dev.recipe.run_deploy import run_deploy_process
    from swift.infer_engine import RequestConfig

    deploy_context = nullcontext(app_config.base_url) if app_config.base_url else run_deploy_process(
        model_config,
        template_config,
        generation_config,
        backend=infer_config.infer_backend,
        engine_args=_engine_args(infer_config.infer_backend, infer_config, rollout_config),
        adapter_mapping=adapter_mapping,
        merge_lora=merge_lora,
        host=deploy_config.host,
        port=deploy_config.port,
        served_model_name=deploy_config.served_model_name,
        owned_by=deploy_config.owned_by,
        api_key=deploy_config.api_key,
        max_logprobs=deploy_config.max_logprobs,
        max_concurrency=deploy_config.max_concurrency,
        log_interval=deploy_config.log_interval,
        request_log_path=deploy_config.request_log_path,
        verbose=deploy_config.verbose,
        ssl_keyfile=deploy_config.ssl_keyfile,
        ssl_certfile=deploy_config.ssl_certfile,
        log_level=deploy_config.log_level)
    model_name = deploy_config.served_model_name
    request_config = RequestConfig(
        max_tokens=generation_config.max_new_tokens,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        repetition_penalty=generation_config.repetition_penalty,
        num_beams=generation_config.num_beams,
        stop=generation_config.stop_words,
        stream=True if generation_config.stream is None else generation_config.stream,
        logprobs=generation_config.logprobs,
        top_logprobs=generation_config.top_logprobs,
        structured_outputs_regex=generation_config.structured_outputs_regex)
    with deploy_context as base_url:
        demo = build_app_ui(
            base_url, model_name, request_config, app_config, default_system=template_config.system)
        concurrency = 1 if infer_config.infer_backend == 'transformers' else 16
        queue_kwargs = ({'concurrency_count': concurrency} if version.parse(gradio.__version__) < version.parse('4')
                        else {'default_concurrency_limit': concurrency})
        demo.queue(**queue_kwargs).launch(
            server_name=app_config.server_name, server_port=app_config.server_port, share=app_config.share)
