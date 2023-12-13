import gradio as gr

from swift.llm import DATASET_MAPPING
from swift.ui.i18n import get_i18n_labels


def dataset():
    get_i18n_labels(i18n)
    with gr.Row():
        dataset = gr.Dropdown(elem_id='dataset',
                              multiselect=True,
                              choices=list(DATASET_MAPPING.keys()), scale=10)
        custom_train_dataset_path = gr.Textbox(elem_id='custom_train_dataset_path', scale=20)
        custom_val_dataset_path = gr.Textbox(elem_id='custom_val_dataset_path', scale=20)
    with gr.Row():
        dataset_test_ratio = gr.Slider(elem_id='dataset_test_ratio', minimum=0.0, maximum=1.0, step=0.05, scale=10)
        train_dataset_sample = gr.Textbox(elem_id='train_dataset_sample', scale=20)
        val_dataset_sample = gr.Textbox(elem_id='val_dataset_sample', scale=20)
        truncation_strategy = gr.Dropdown(elem_id='truncation_strategy', scale=10)


i18n = {
    "dataset": {
        "label": {
            "zh": "数据集名称",
            "en": "Dataset Code"
        },
        "info": {
            "zh": "选择训练的数据集，支持复选",
            "en": "The dataset(s) to train the models"
        }
    },
    "custom_train_dataset_path": {
        "label": {
            "zh": "自定义训练数据集路径",
            "en": "Custom train dataset path"
        },
        "info": {
            "zh": "输入自定义的训练数据集路径，逗号分隔",
            "en": "Extra train files, split by comma"
        }
    },
    "custom_val_dataset_path": {
        "label": {
            "zh": "自定义校验数据集路径",
            "en": "Custom val dataset path"
        },
        "info": {
            "zh": "输入自定义的校验数据集路径，逗号分隔",
            "en": "Extra val files, split by comma"
        }
    },
    "dataset_test_ratio": {
        "label": {
            "zh": "数据集拆分比例",
            "en": "The split ratio of datasets"
        },
        "info": {
            "zh": "将数据集按照比例拆分为训练集和验证集",
            "en": "Split the datasets by this ratio to train/val"
        }
    },
    "train_dataset_sample": {
        "label": {
            "zh": "训练集采样数量",
            "en": "The sample size from the train dataset"
        },
        "info": {
            "zh": "从训练集中采样一定行数进行训练",
            "en": "Train with the sample size from the dataset"
        }
    },
    "val_dataset_sample": {
        "label": {
            "zh": "验证集采样数量",
            "en": "The sample size from the val dataset"
        },
        "info": {
            "zh": "从验证集中采样一定行数进行训练",
            "en": "Validate with the sample size from the dataset"
        }
    },
    "truncation_strategy": {
        "label": {
            "zh": "数据集超长策略",
            "en": "Dataset truncation strategy"
        },
        "info": {
            "zh": "如果token超长该如何处理",
            "en": "How to deal with the rows exceed the max length"
        }
    }
}