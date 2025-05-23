# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Type

import gradio as gr

from swift.ui.base import BaseUI


class Rollout(BaseUI):
    group = 'llm_grpo'

    locale_dict = {
        'num_generations': {
            'label': {
                'zh': '采样数量',
                'en': 'Number of samples'
            },
            'info': {
                'zh': '每个prompt采样的数量，即论文中的G值',
                'en': 'The number of samples for each prompt, that is, the G value in the paper'
            }
        },
        'max_completion_length': {
            'label': {
                'zh': '最大生成长度',
                'en': 'Max completion length'
            },
            'info': {
                'zh': 'GRPO算法中的最大生成长度',
                'en': 'Maximum generation length in GRPO algorithm'
            }
        },
        'dynamic_sample': {
            'label': {
                'zh': '动态采样',
                'en': 'Dynamic Sampling'
            },
            'info': {
                'zh': '筛除group内奖励标准差为0的数据，额外采样新数据',
                'en': 'Filter out data with a reward standard deviation of 0 within the group and sample new data'
            }
        },
        'max_resample_times': {
            'label': {
                'zh': '最大重采样次数',
                'en': 'Maximum number of resampling times'
            },
            'info': {
                'zh': 'dynamic_sample设置下限制重采样次数',
                'en': 'Limit the number of resampling times when dynamic_sample is set'
            }
        },
        'overlong_filter': {
            'label': {
                'zh': '跳过超长样本',
                'en': 'Skip overlong samples'
            },
            'info': {
                'zh': '跳过超长截断的样本，不参与loss计算',
                'en': 'Skip overlong truncated samples and exclude them from loss calculation'
            }
        },
        'log_completions': {
            'label': {
                'zh': '记录生成内容',
                'en': 'Record generated content'
            },
            'info': {
                'zh': '是否记录训练中的模型生成内容',
                'en': 'Whether to record the model generation content during training'
            }
        },
        'async_generate': {
            'label': {
                'zh': '异步生成',
                'en': 'Async generate'
            },
            'info': {
                'zh': '异步rollout以提高训练速度',
                'en': 'Asynchronous rollout to increase training speed'
            }
        },
        'temperature': {
            'label': {
                'zh': 'temperature',
                'en': 'temperature'
            },
        },
        'top_k': {
            'label': {
                'zh': 'top_k',
                'en': 'top_k'
            },
        },
        'top_p': {
            'label': {
                'zh': 'top_p',
                'en': 'top_p'
            },
        },
        'repetition_penalty': {
            'label': {
                'zh': '重复惩罚项',
                'en': 'repetition penalty'
            },
        },
        'use_vllm': {
            'label': {
                'zh': '使用vLLM',
                'en': 'Using vLLM'
            },
            'info': {
                'zh': '是否使用vLLM作为GRPO生成的infer_backend',
                'en': 'Whether to use vLLM as the infer_backend of generation by GRPO'
            }
        },
        'vllm_mode': {
            'label': {
                'zh': 'vLLM集成模式',
                'en': 'vLLM Integration Mode'
            },
            'info': {
                'zh':
                'server模式使用`swift rollout`拉起的vLLM服务进行采样;colocate模式使用程序内部署的vllm',
                'en':
                'Server mode uses the vLLM server deployed by swift rollout for sampling,'
                ' colocate mode uses vllm deployed in the program'
            }
        },
        'vllm_enable_prefix_caching': {
            'label': {
                'zh': '开启前缀缓存',
                'en': 'Enable prefix cache'
            },
            'info': {
                'zh': 'vllm透传参数',
                'en': 'vllm transparent transmission parameters'
            }
        },
        'vllm_gpu_memory_utilization': {
            'label': {
                'zh': 'GPU显存利用率',
                'en': 'Gpu memory utilization'
            },
            'info': {
                'zh': 'vllm透传参数',
                'en': 'vllm transparent transmission parameters'
            }
        },
        'vllm_tensor_parallel_size': {
            'label': {
                'zh': '张量并行大小',
                'en': 'Tensor parallel size'
            },
            'info': {
                'zh': 'vllm透传参数',
                'en': 'vllm transparent transmission parameters'
            }
        },
        'vllm_max_model_len': {
            'label': {
                'zh': '模型支持的最大长度',
                'en': 'max model len'
            },
            'info': {
                'zh': 'vllm透传参数',
                'en': 'vllm transparent transmission parameters'
            }
        },
        'sleep_level': {
            'label': {
                'zh': 'sleep level',
                'en': 'sleep level'
            },
            'info': {
                'zh': '训练时释放vLLM显存',
                'en': 'Release vLLM memory during training'
            }
        },
        'vllm_server_host': {
            'label': {
                'zh': '服务主机',
                'en': 'vllm server host'
            },
            'info': {
                'zh': 'vllm服务host',
                'en': 'The actual model id or model path'
            }
        },
        'vllm_server_port': {
            'label': {
                'zh': '服务端口号',
                'en': 'vllm server port'
            },
            'info': {
                'zh': 'vllm服务端口号',
                'en': 'The actual model id or model path'
            }
        },
        'vllm_server_timeout': {
            'label': {
                'zh': '服务超时时间',
                'en': 'Server timeout'
            },
            'info': {
                'zh': '连接vLLM server的超时时间',
                'en': 'Timeout for connecting to vLLM server'
            }
        },
        'offload_model': {
            'label': {
                'zh': '卸载模型',
                'en': 'offload model'
            },
            'info': {
                'zh': '是否在vLLM/LMDeploy推理时offload模型',
                'en': 'Whether to offload the model during vLLM/LMDeploy inference'
            }
        },
        'offload_optimizer': {
            'label': {
                'zh': '卸载优化器',
                'en': 'offload optimizer'
            },
            'info': {
                'zh': '是否在vLLM/LMDeploy推理时offload optimizer参数',
                'en': 'Whether to offload optimizer parameters during vLLM/LMDeploy inference'
            }
        },
        'gc_collect_after_offload': {
            'label': {
                'zh': 'offload结束时gc',
                'en': 'gc collect after offload'
            },
            'info': {
                'zh': '是否在offload结束时进行gc',
                'en': 'Whether to perform GC at the end of offload'
            }
        },
        'colocate_param': {
            'label': {
                'zh': 'colocate模式参数',
                'en': 'colocate mode parameters'
            }
        },
        'server_param': {
            'label': {
                'zh': 'server模式参数',
                'en': 'server mode parameters'
            }
        },
        'rollout_param': {
            'label': {
                'zh': 'rollout设置',
                'en': 'Rollout settings'
            }
        }
    }

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.Accordion(elem_id='rollout_param', open=True):
            with gr.Row():
                gr.Slider(elem_id='num_generations', minimum=1, maximum=64, step=1, value=8, scale=4)
                gr.Textbox(elem_id='max_completion_length', lines=1, value='512', scale=4)
                gr.Checkbox(elem_id='dynamic_sample', scale=4)
                gr.Slider(elem_id='max_resample_times', minimum=1, maximum=16, step=1, value=3, scale=4)
                gr.Checkbox(elem_id='overlong_filter', scale=4)
                gr.Checkbox(elem_id='log_completions', scale=4)

            with gr.Row():
                gr.Slider(elem_id='temperature', minimum=0.0, maximum=10, step=0.1, value=1.0)
                gr.Slider(elem_id='top_k', minimum=1, maximum=100, step=5, value=20)
                gr.Slider(elem_id='top_p', minimum=0.0, maximum=1.0, step=0.05, value=1.0)
                gr.Slider(elem_id='repetition_penalty', minimum=0.0, maximum=10, step=0.05, value=1.05)

            with gr.Row():
                gr.Checkbox(elem_id='use_vllm', scale=4)
                gr.Dropdown(elem_id='vllm_mode', choices=['colocate', 'server'], scale=4)
                gr.Checkbox(elem_id='offload_model', scale=4)
                gr.Checkbox(elem_id='offload_optimizer', scale=4)
                gr.Checkbox(elem_id='gc_collect_after_offload', scale=4)

            with gr.Accordion(elem_id='colocate_param', open=True):
                with gr.Row():
                    gr.Checkbox(elem_id='vllm_enable_prefix_caching', scale=4)
                    gr.Dropdown(elem_id='sleep_level', choices=[0, 1], value=0, scale=4)
                    gr.Textbox(elem_id='vllm_gpu_memory_utilization', lines=1, value='0.5', scale=4)
                    gr.Textbox(elem_id='vllm_tensor_parallel_size', lines=1, value='1', scale=4)
                    gr.Textbox(elem_id='vllm_max_model_len', lines=1, value='', scale=4)
            with gr.Accordion(elem_id='server_param', open=True):
                with gr.Row():
                    gr.Checkbox(elem_id='async_generate', scale=1)
                    gr.Textbox(elem_id='vllm_server_host', scale=4)
                    gr.Textbox(elem_id='vllm_server_port', lines=1, scale=4)
                    gr.Textbox(elem_id='vllm_server_timeout', lines=1, scale=4, value=120)
