import os
import time
from dataclasses import fields
import json
import gradio
import gradio as gr
import torch
from swift.llm import SftArguments
from swift.ui.element import elements, components
from swift.ui.i18n import get_i18n_labels
from swift.ui.llm.advanced import advanced
from swift.ui.llm.dataset import dataset
from swift.ui.llm.hyper import hyper
from swift.ui.llm.lora import lora
from swift.ui.llm.model import model
from swift.ui.llm.save import save
from swift.ui.llm.self_cog import self_cognition
from swift.ui.llm.runtime import runtime


def llm_train():
    gpu_count = 0
    default_device = 'cpu'
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        default_device = '0'
    get_i18n_labels(i18n)
    with gr.Blocks():
        model()
        dataset()
        runtime()
        with gr.Row():
            gr.Dropdown(elem_id='sft_type', scale=4)
            gr.Textbox(elem_id='seed', scale=4)
            gr.Dropdown(elem_id='dtype', scale=4)
            gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
            gr.Slider(elem_id='neftune_alpha', minimum=0.0, maximum=1.0, step=0.05, scale=4)
        with gr.Row():
            gr.Dropdown(
                elem_id='gpu_id', multiselect=True, choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                value=default_device,
                scale=8)
            gr.Textbox(elem_id='gpu_memory_fraction', value='1.0', scale=4)
            gr.Button(elem_id='submit', scale=4, variant='primary')

        save()
        lora()
        hyper()
        self_cognition()
        advanced()

        elements['submit'].click(
            train,
            [],
            [elements['running_cmd'], elements['logging_dir'], elements['runtime_tab']],
            show_progress=True
        )


def train():
    ignore_elements = ('model_type', 'logging_dir')
    args = fields(SftArguments)
    args = {arg.name: arg.type for arg in args}
    kwargs = {}
    more_params = getattr(elements['more_params'], 'last_value', None)
    if more_params:
        more_params = json.loads(more_params)
    else:
        more_params = {}

    for e in elements:
        if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'last_value', None) \
            and e not in ignore_elements:
            kwargs[e] = elements[e].last_value
    kwargs.update(more_params)
    sft_args = SftArguments(**kwargs)
    params = ''

    for e in elements:
        if e in args and getattr(elements[e], 'changed', False) and getattr(elements[e], 'last_value', None) and e != 'model_type':
            params += f'--{e} "{elements[e].last_value}" '
    for key, param in more_params.items():
        params += f'--{key} "{param}" '
    ddp_param = ''
    if elements['use_ddp'] and getattr(elements["gpu_id"], 'last_value', None):
        ddp_param = f'NPROC_PER_NODE={len(elements["gpu_id"].last_value)}'
    if hasattr(elements["gpu_id"], 'last_value') and isinstance(elements["gpu_id"].last_value, list):
        assert(len(elements["gpu_id"].last_value) == 1 or 'cpu' not in elements["gpu_id"].last_value)
        gpus = ','.join(elements["gpu_id"].last_value)
    else:
        gpus = '0'
    cuda_param = ''
    if gpus != 'cpu':
        cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'
    os.makedirs(sft_args.logging_dir, exist_ok=True)
    log_file = os.path.join(sft_args.logging_dir, "run.log")
    run_command = f'{cuda_param} {ddp_param} nohup swift sft {params} > {log_file} 2>&1 &'
    os.system(run_command)
    time.sleep(1) # to make sure the log file has been created.
    gradio.Info(components['submit_alert']['value'])
    return run_command, sft_args.logging_dir, gr.update(visible=True)


i18n = {
    "submit_alert": {
        "value": {
            "zh": "任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程",
            "en": "Task started, please check the tensorboard or log file, "
                  "closing this page does not affect training"
        }
    },
    "submit": {
        "value": {
            "zh": "开始训练",
            "en": "Begin"
        }
    },
    "gpu_id": {
        "label": {
            "zh": "选择可用GPU",
            "en": "Choose GPU"
        },
        "info": {
            "zh": "选择训练使用的GPU号，如CUDA不可用只能选择CPU",
            "en": "Select GPU to train"
        }
    },
    "gpu_memory_fraction": {
        "label": {
            "zh": "GPU显存限制",
            "en": "GPU memory fraction"
        },
        "info": {
            "zh": "设置使用显存的比例，一般用于显存测试",
            "en": "Set the memory fraction ratio of GPU, usually used in memory test"
        }
    },
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