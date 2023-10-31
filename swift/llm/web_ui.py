from typing import Tuple

from .infer import prepare_model_template
from .utils import History, InferArguments, inference_stream


def clear_session() -> History:
    return []


def gradio_generation_demo(args: InferArguments) -> None:
    import gradio as gr
    model, template = prepare_model_template(args)

    def model_generation(query: str) -> str:
        gen = inference_stream(
            model, template, query, None, skip_special_tokens=True)
        for response, _ in gen:
            yield response

    model_name = args.model_type.title()

    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(lines=16, label='Input', max_lines=16)
            with gr.Column(scale=1):
                output_box = gr.Textbox(lines=16, label='Output', max_lines=16)
        send = gr.Button('🚀 发送')
        send.click(model_generation, inputs=[input_box], outputs=[output_box])
    demo.queue().launch(height=1000)


def gradio_chat_demo(args: InferArguments, history_length: int = 10) -> None:
    import gradio as gr
    model, template = prepare_model_template(args)

    def model_chat(query: str, history: History) -> Tuple[str, History]:
        old_history = history[:-history_length]
        history = history[-history_length:]
        gen = inference_stream(
            model, template, query, history, skip_special_tokens=True)
        for _, history in gen:
            total_history = old_history + history
            yield '', total_history

    model_name = args.model_type.title()
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><font size=8>{model_name} Bot</center>')

        chatbot = gr.Chatbot(label=f'{model_name}')
        message = gr.Textbox(lines=3)
        with gr.Row():
            clear_history = gr.Button('🧹 清除历史对话')
            send = gr.Button('🚀 发送')
        send.click(
            model_chat, inputs=[message, chatbot], outputs=[message, chatbot])
        clear_history.click(
            fn=clear_session, inputs=[], outputs=[chatbot], queue=False)
    demo.queue().launch(height=1000)
