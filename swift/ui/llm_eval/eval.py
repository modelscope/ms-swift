import os.path
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI
from swift.utils import get_logger

logger = get_logger()


class Eval(BaseUI):

    group = 'llm_eval'

    locale_dict = {
        'name': {
            'label': {
                'zh': '评测名称',
                'en': 'Evaluation name'
            },
            'info': {
                'zh': '支持英文字母、下划线、横线和数字',
                'en': 'Support characters, underscores, hyphens and numbers'
            }
        },
        'eval_dataset': {
            'label': {
                'zh': '评测数据集',
                'en': 'Evaluation dataset'
            },
            'info': {
                'zh': '选择评测数据集，支持多选',
                'en': 'Select eval dataset, multiple datasets supported'
            }
        },
        'eval_few_shot': {
            'label': {
                'zh': 'prompt的few-shot',
                'en': 'The few-shot for the prompt'
            },
            'info': {
                'zh': 'Few-shot数量在评测集中有默认设置，可以不填',
                'en': 'Few-shot numbers have default values in different datasets'
            }
        },
        'eval_limit': {
            'label': {
                'zh': '评测数据个数',
                'en': 'Eval numbers for each dataset'
            },
            'info': {
                'zh': '每个评测集的取样数',
                'en': 'Number of rows sampled from each dataset'
            }
        },
        'eval_use_cache': {
            'label': {
                'zh': '使用缓存',
                'en': 'Use eval cache'
            },
            'info': {
                'zh': '如果name指定的评测已经存在，则可以使用已有缓存',
                'en': 'If the evaluation results of the name exists, you may use cache.'
            }
        },
        'custom_eval_config': {
            'label': {
                'zh': '自定义数据集评测配置',
                'en': 'Custom eval config'
            },
            'info': {
                'zh': '可以使用该配置评测自己的数据集，详见github文档的评测部分',
                'en': 'Use this config to eval your own datasets, check the docs in github for details'
            }
        },
        'eval_url': {
            'label': {
                'zh': '评测链接',
                'en': 'The eval url'
            },
            'info': {
                'zh':
                'OpenAI样式的评测链接(如：http://localhost:8080/v1)，用于评测接口（模型类型输入为实际模型类型）',
                'en':
                'The OpenAI style link(like: http://localhost:8080/v1) for '
                'evaluation(Input actual model type into model_type)'
            }
        },
        'eval_token': {
            'label': {
                'zh': 'Url token',
                'en': 'The url token'
            },
        },
        'eval_is_chat_model': {
            'label': {
                'zh': '接口是chat模型',
                'en': 'Chat model'
            },
            'info': {
                'zh': '评测接口是否是Chat模型',
                'en': 'The eval url is a chat model or not'
            }
        },
        'infer_backend': {
            'label': {
                'zh': '推理框架',
                'en': 'Infer backend'
            },
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        try:
            from evalscope.backend.opencompass import OpenCompassBackendManager
            from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
            eval_dataset_list = (
                OpenCompassBackendManager.list_datasets() + VLMEvalKitBackendManager.list_supported_datasets())
        except Exception as e:
            logger.error(e)
            eval_dataset_list = [
                'AX_b', 'cmb', 'winogrande', 'mmlu', 'afqmc', 'COPA', 'commonsenseqa', 'CMRC', 'lcsts', 'nq',
                'ocnli_fc', 'math', 'mbpp', 'DRCD', 'TheoremQA', 'CB', 'ReCoRD', 'lambada', 'tnews', 'flores',
                'humaneval', 'AX_g', 'ceval', 'bbh', 'BoolQ', 'MultiRC', 'piqa', 'csl', 'ARC_c', 'agieval', 'cmnli',
                'strategyqa', 'gsm8k', 'summedits', 'eprstmt', 'WiC', 'cluewsc', 'Xsum', 'ocnli', 'triviaqa',
                'hellaswag', 'race', 'bustm', 'RTE', 'C3', 'GaokaoBench', 'storycloze', 'ARC_e', 'siqa', 'obqa', 'WSC',
                'chid', 'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN',
                'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11',
                'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11',
                'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI',
                'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench',
                'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE',
                'TextVQA_VAL', 'DocVQA_VAL', 'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision',
                'MathVision_MINI', 'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet',
                'MTVQA_TEST', 'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL',
                'VCR_EN_HARD_500', 'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100',
                'VCR_ZH_EASY_ALL', 'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video',
                'Video-MME', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench',
                'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11',
                'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL',
                'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL',
                'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS',
                'BLINK'
            ]

        with gr.Row():
            gr.Textbox(elem_id='name', scale=20)
            gr.Dropdown(
                elem_id='eval_dataset',
                is_list=True,
                choices=eval_dataset_list,
                multiselect=True,
                allow_custom_value=True,
                scale=20)
            gr.Textbox(elem_id='eval_few_shot', scale=20)
            gr.Textbox(elem_id='eval_limit', scale=20)
            gr.Checkbox(elem_id='eval_use_cache', scale=20)
            gr.Dropdown(elem_id='infer_backend', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='custom_eval_config', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='eval_url', scale=20)
            gr.Textbox(elem_id='eval_token', scale=20)
            gr.Checkbox(elem_id='eval_is_chat_model', scale=20)
