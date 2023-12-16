import json
import os
from dataclasses import fields

import gradio as gr
import torch

from swift.llm import InferArguments, prepare_model_template, limit_history_length, inference_stream
from swift.ui.element import get_elements_by_group
from swift.ui.i18n import add_locale_labels
from swift.ui.llm_infer.generate import generate

elements = get_elements_by_group('llm_train')

def llm_infer():
    gpu_count = 0
    default_device = 'cpu'
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        default_device = '0'
    add_locale_labels(locale_dict, 'llm_infer')
    with gr.Blocks():
        model_and_template = gr.State([])
        model()
        generate()
        gr.Dropdown(
            elem_id='gpu_id',
            multiselect=True,
            choices=[str(i) for i in range(gpu_count)] + ['cpu'],
            value=default_device,
            scale=8)
        chatbot = gr.Chatbot(elem_id='chatbot', lines=10, elem_classes="control-height")
        prompt = gr.Textbox(elem_id='prompt', lines=2)

        with gr.Row():
            clear_history = gr.Button(elem_id='clear_history')
            submit = gr.Button(elem_id='submit')

        submit.click(generate_chat,
                     inputs=[model_and_template, prompt, chatbot, elements['max_new_tokens']],
                     outputs=[prompt, chatbot])
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[prompt, chatbot],
                            queue=False)
        elements['load_checkpoint'].click(
            prepare_checkpoint, [], [model_and_template, prompt],
            show_progress=True)
        elements['load_checkpoint'].click(clear_session,
                                          inputs=[],
                                          outputs=[prompt, chatbot])


def prepare_checkpoint():
    global model, tokenizer, template
    args = fields(InferArguments)
    args = {arg.name: arg.type for arg in args}
    kwargs = {}
    more_params = getattr(elements['more_params'], 'last_value', None)
    if more_params:
        more_params = json.loads(more_params)
    else:
        more_params = {}

    for e in elements:
        if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'last_value', None):
            kwargs[e] = elements[e].last_value
    kwargs.update(more_params)

    devices = getattr(elements['gpu_id'], 'last_value',
                      ' '.join(elements['gpu_id'].value)).split(' ')
    devices = [d for d in devices if d]
    assert (len(devices) == 1 or 'cpu' not in devices)
    gpus = ','.join(devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    args = InferArguments(**kwargs)
    model, template = prepare_model_template(args)
    return [model, template], gr.update(interactive=True)


def clear_session():
    return '', None


def generate_chat(model_and_template, prompt: str, history, max_new_tokens):
    model, template = model_and_template
    if not elements['template_type'].endswith('generation'):
        old_history, history = limit_history_length(template, prompt, history,
                                                    max_new_tokens)
    else:
        old_history = []
        history = []
    gen = inference_stream(model, template, prompt, history)
    for _, history in gen:
        total_history = old_history + history
        yield '', total_history


locale_dict = {
    'load_alert': {
        'value': {
            'zh':
            '加载模型中，请等待',
            'en':
            'Start to load model, please wait'
        }
    },
    'load_checkpoint': {
        'value': {
            'zh':
                '加载模型',
            'en':
                'Load model'
        }
    },
    'chatbot': {
        'label': {
            'zh': '对话框',
            'en': 'Chat bot'
        },
    },
    'prompt': {
        'label': {
            'zh': '请输入：',
            'en': 'Input:'
        },
    },
    'clear_history': {
        'label': {
            'zh': '清除对话信息',
            'en': 'Clear history'
        },
    },
    'submit': {
        'label': {
            'zh': '🚀 发送',
            'en': '🚀 Send'
        },
    },
    'gpu_id': {
        'label': {
            'zh': '选择可用GPU',
            'en': 'Choose GPU'
        },
        'info': {
            'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
            'en': 'Select GPU to train'
        }
    },
}
