import gradio as gr
from swift.ui.llm_train import llm_train

with gr.Blocks() as app:
    with gr.Tabs():
        with gr.TabItem(elem_id='llm_train'):
            llm_train()
        # with gr.TabItem(page_content['llm_inference']):
        #     llm_inference()
        # with gr.TabItem(page_content['sd_train']):
        #     sd_train()
        # with gr.TabItem(page_content['sd_inference']):
        #     sd_inference()
        # with gr.TabItem(page_content['llm_deploy']):
        #     llm_deploy()

app.queue().launch(height=800, share=False)

