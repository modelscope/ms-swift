import gradio as gr

from swift.ui.i18n import get_i18n_labels
from swift.ui.llm.advanced import advanced
from swift.ui.llm.dataset import dataset
from swift.ui.llm.hyper import hyper
from swift.ui.llm.lora import lora
from swift.ui.llm.model import model
from swift.ui.llm.save import save
from swift.ui.llm.self_cog import self_cognition


def llm_train():
    get_i18n_labels(i18n)
    with gr.Blocks():
        model()
        dataset()
        with gr.Row():
            gr.Dropdown(elem_id='sft_type', scale=4)
            gr.Textbox(elem_id='seed', scale=4)
            gr.Dropdown(elem_id='dtype', scale=4)
            gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
            gr.Slider(elem_id='neftune_alpha', minimum=0.0, maximum=1.0, step=0.05, scale=4)
        save()
        lora()
        hyper()
        self_cognition()
        advanced()


i18n = {
    "sft_type": {
        "label": {
            "zh": "训练方式",
            "en": "Train type"
        },
        "info": {
            "zh": "选择训练的方式",
            "en": "Select the training type"
        }
    },
    "seed": {
        "label": {
            "zh": "随机数种子",
            "en": "Seed"
        },
        "info": {
            "zh": "选择随机数种子",
            "en": "Select a random seed"
        }
    },
    "dtype": {
        "label": {
            "zh": "训练精度",
            "en": "Training Precision"
        },
        "info": {
            "zh": "选择训练精度",
            "en": "Select the training precision"
        }
    },
    "use_ddp": {
        "label": {
            "zh": "使用DDP",
            "en": "Use DDP"
        },
        "info": {
            "zh": "是否使用数据并行训练",
            "en": "Use Distributed Data Parallel to train"
        }
    },
    "neftune_alpha": {
        "label": {
            "zh": "neftune_alpha",
            "en": "neftune_alpha"
        },
        "info": {
            "zh": "使用neftune提升训练效果",
            "en": "Use neftune to improve performance"
        }
    }
}