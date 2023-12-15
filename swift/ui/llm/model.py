import gradio as gr

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
from swift.ui.i18n import get_i18n_labels


def model():
    get_i18n_labels(i18n)
    with gr.Row():
        model_type = gr.Dropdown(
            elem_id='model_type', choices=list(MODEL_MAPPING.keys()), scale=20)
        model_id_or_path = gr.Textbox(
            elem_id='model_id_or_path', lines=1, scale=20)
        template_type = gr.Dropdown(
            elem_id='template_type',
            choices=list(TEMPLATE_MAPPING.keys()) + ['AUTO'],
            scale=20)
    with gr.Row():
        system = gr.Textbox(elem_id='system', lines=1, scale=20)

    def update_input_model(choice):
        return MODEL_MAPPING[choice]['model_id_or_path'], \
            TEMPLATE_MAPPING[MODEL_MAPPING[choice]['template']].default_system, \
            MODEL_MAPPING[choice]['template']

    model_type.change(
        update_input_model,
        inputs=[model_type],
        outputs=[model_id_or_path, system, template_type])


i18n = {
    'model_type': {
        'label': {
            'zh': '选择模型',
            'en': 'Select Model'
        },
        'info': {
            'zh': 'SWIFT已支持的模型名称',
            'en': 'Base model supported by SWIFT'
        }
    },
    'model_id_or_path': {
        'label': {
            'zh': '模型id或路径',
            'en': 'Model id or path'
        },
        'info': {
            'zh': '实际的模型id',
            'en': 'The actual model id or model path'
        }
    },
    'template_type': {
        'label': {
            'zh': '模型Prompt模板类型',
            'en': 'Prompt template type'
        },
        'info': {
            'zh': '选择匹配模型的Prompt模板',
            'en': 'Choose the template type of the model'
        }
    },
    'system': {
        'label': {
            'zh': 'system字段',
            'en': 'system'
        },
        'info': {
            'zh': '选择system字段的内容',
            'en': 'Choose the content of the system field'
        }
    },
}
