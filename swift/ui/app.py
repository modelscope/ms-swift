# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import fields
from functools import partial
from typing import List, Union

import gradio as gr
from packaging import version
from transformers.utils import strtobool

import swift
from swift.llm import DeployArguments, EvalArguments, ExportArguments, RLHFArguments, SwiftPipeline, WebUIArguments
from swift.ui.llm_eval.llm_eval import LLMEval
from swift.ui.llm_export.llm_export import LLMExport
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain

locale_dict = {
    'title': {
        'zh': '🚀SWIFT: 轻量级大模型训练推理框架',
        'en': '🚀SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning and Inference'
    },
    'sub_title': {
        'zh':
        '请查看 <a href=\"https://github.com/modelscope/swift/tree/main/docs/source\" target=\"_blank\">'
        'SWIFT 文档</a>来查看更多功能，使用SWIFT_UI_LANG=en环境变量来切换英文界面',
        'en':
        'Please check <a href=\"https://github.com/modelscope/swift/tree/main/docs/source_en\" target=\"_blank\">'
        'SWIFT Documentation</a> for more usages, Use SWIFT_UI_LANG=zh variable to switch to Chinese UI',
    },
    'star_beggar': {
        'zh':
        '喜欢<a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>就动动手指给我们加个star吧🥺 ',
        'en':
        'If you like <a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>, '
        'please take a few seconds to star us🥺 '
    },
}


class SwiftWebUI(SwiftPipeline):

    args_class = WebUIArguments
    args: args_class

    def run(self):
        lang = os.environ.get('SWIFT_UI_LANG') or self.args.lang
        share_env = os.environ.get('WEBUI_SHARE')
        share = strtobool(share_env) if share_env else self.args.share
        server = os.environ.get('WEBUI_SERVER') or self.args.host
        port_env = os.environ.get('WEBUI_PORT')
        port = int(port_env) if port_env else self.args.port
        is_gradio_app = self.args.model or self.args.ckpt_dir
        LLMTrain.set_lang(lang)
        LLMInfer.set_lang(lang)
        LLMExport.set_lang(lang)
        LLMEval.set_lang(lang)
        with gr.Blocks(title='SWIFT WebUI') as app:
            if is_gradio_app:
                gr.HTML(f'<h1><center>{self.args.studio_title}</center></h1>')
            else:
                try:
                    _version = swift.__version__
                except AttributeError:
                    _version = ''
                gr.HTML(f"<h1><center>{locale_dict['title'][lang]}({_version})</center></h1>")
                gr.HTML(f"<h3><center>{locale_dict['sub_title'][lang]}</center></h3>")
            with gr.Tabs():
                if is_gradio_app:
                    if self.args.ckpt_dir:
                        self.args.model = self.args.ckpt_dir
                    for f in fields(self.args):
                        if getattr(self.args, f.name):
                            LLMInfer.default_dict[f.name] = getattr(self.args, f.name)
                    LLMInfer.is_gradio_app = True
                    LLMInfer.is_multimodal = self.args.model_meta.is_multimodal
                    LLMInfer.build_ui(LLMInfer)
                else:
                    LLMTrain.build_ui(LLMTrain)
                    LLMInfer.build_ui(LLMInfer)
                    LLMExport.build_ui(LLMExport)
                    LLMEval.build_ui(LLMEval)

            concurrent = {}
            if version.parse(gr.__version__) < version.parse('4.0.0'):
                concurrent = {'concurrency_count': 5}
            if is_gradio_app:
                from swift.utils import find_free_port
                LLMInfer.element('port').value = str(find_free_port())
                app.load(LLMInfer.deploy_model, list(LLMInfer.valid_elements().values()),
                         [LLMInfer.element('runtime_tab'),
                          LLMInfer.element('running_tasks')])
            else:
                app.load(
                    partial(LLMTrain.update_input_model, arg_cls=RLHFArguments),
                    inputs=[LLMTrain.element('model')],
                    outputs=[LLMTrain.element('train_record')] + list(LLMTrain.valid_elements().values()))
                app.load(
                    partial(LLMInfer.update_input_model, arg_cls=DeployArguments, has_record=False),
                    inputs=[LLMInfer.element('model')],
                    outputs=list(LLMInfer.valid_elements().values()))
                app.load(
                    partial(LLMExport.update_input_model, arg_cls=ExportArguments, has_record=False),
                    inputs=[LLMExport.element('model')],
                    outputs=list(LLMExport.valid_elements().values()))
                app.load(
                    partial(LLMEval.update_input_model, arg_cls=EvalArguments, has_record=False),
                    inputs=[LLMEval.element('model')],
                    outputs=list(LLMEval.valid_elements().values()))
        app.queue(**concurrent).launch(server_name=server, inbrowser=True, server_port=port, height=800, share=share)


def webui_main(args: Union[List[str], WebUIArguments, None] = None):
    return SwiftWebUI(args).main()
