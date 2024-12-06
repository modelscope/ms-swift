# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI
from swift.utils import get_logger

logger = get_logger()


class Eval(BaseUI):

    group = 'llm_eval'

    locale_dict = {
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
        'eval_output_dir': {
            'label': {
                'zh': '评测输出目录',
                'en': 'Eval output dir'
            },
            'info': {
                'zh': '评测结果的输出目录',
                'en': 'The dir to save the eval results'
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
                'OpenAI样式的评测链接(如：http://localhost:8080/v1/chat/completions)，用于评测接口（模型类型输入为实际模型类型）',
                'en':
                'The OpenAI style link(like: http://localhost:8080/v1/chat/completions) for '
                'evaluation(Input actual model type into model_type)'
            }
        },
        'api_key': {
            'label': {
                'zh': '接口token',
                'en': 'The url token'
            },
            'info': {
                'zh': 'eval_url的token',
                'en': 'The token used with eval_url'
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
            logger.warn('If you encounter an error message👆🏻👆🏻👆🏻 of `.env` file, please ignore.')
        except Exception as e:
            logger.warn(e)
            logger.warn(
                ('The error message 👆🏻👆🏻👆🏻above will have no bad effects, '
                 'only means evalscope is not installed, and default eval datasets will be listed in the web-ui.'))
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
            gr.Dropdown(
                elem_id='eval_dataset',
                is_list=True,
                choices=eval_dataset_list,
                multiselect=True,
                allow_custom_value=True,
                scale=20)
            gr.Textbox(elem_id='eval_limit', scale=20)
            gr.Dropdown(elem_id='infer_backend', scale=20)
        with gr.Row():
            gr.Textbox(elem_id='custom_eval_config', scale=20)
            gr.Textbox(elem_id='eval_output_dir', scale=20)
            gr.Textbox(elem_id='eval_url', scale=20)
            gr.Textbox(elem_id='api_key', scale=20)
