import gradio as gr

from swift.ui.i18n import get_i18n_labels
from swift.ui.llm.llm_train import llm_train

i18n = {
    'llm_train': {
        'label': {
            'zh': 'LLM训练',
            'en': 'LLM Training',
        }
    }
}

with gr.Blocks() as app:
    get_i18n_labels(i18n)
    with gr.Tabs():
        with gr.TabItem(elem_id='llm_train', label=''):
            llm_train()

app.queue().launch(height=800, share=False)
