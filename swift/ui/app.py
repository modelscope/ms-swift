import os

import gradio as gr
from packaging import version

from swift.ui.base import all_langs
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain

lang = os.environ.get('SWIFT_UI_LANG', all_langs[0])

locale_dict = {
    'title': {
        'zh': '🚀SWIFT: 轻量级大模型训练推理框架',
        'en': '🚀SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning and Inference'
    },
    'sub_title': {
        'zh':
        '请查看 <a href=\"https://github.com/modelscope/swift/tree/main/docs/source\" target=\"_blank\">'
        'SWIFT 文档</a>来查看更多功能',
        'en':
        'Please check <a href=\"https://github.com/modelscope/swift/tree/main/docs/source_en\" target=\"_blank\">'
        'SWIFT Documentation</a> for more usages',
    },
    'star_beggar': {
        'zh':
        '喜欢<a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>就动动手指给我们加个star吧🥺 ',
        'en':
        'If you like <a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>, '
        'please take a few seconds to star us🥺 '
    },
}


def run_ui():
    LLMTrain.set_lang(lang)
    LLMInfer.set_lang(lang)
    with gr.Blocks(title='SWIFT WebUI') as app:
        gr.HTML(f"<h1><center>{locale_dict['title'][lang]}</center></h1>")
        gr.HTML(f"<h3><center>{locale_dict['sub_title'][lang]}</center></h3>")
        gr.HTML(f"<h3><center>{locale_dict['star_beggar'][lang]}</center></h3>")
        with gr.Tabs():
            LLMTrain.build_ui(LLMTrain)
            LLMInfer.build_ui(LLMInfer)

    port = os.environ.get('WEBUI_PORT', None)
    concurrent = {}
    if version.parse(gr.__version__) < version.parse('4.0.0'):
        concurrent = {'concurrency_count': 5}
    app.queue(**concurrent).launch(
        server_name=os.environ.get('WEBUI_SERVER', None),
        server_port=port if port is None else int(port),
        height=800,
        share=bool(int(os.environ.get('WEBUI_SHARE', '0'))))
