import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['SWIFT_DEBUG'] = '1'


def test_deepseek_janus_pro_gene():
    from swift.llm import infer_main, InferArguments
    args = InferArguments(model='deepseek-ai/Janus-Pro-1B', infer_backend='pt')
    infer_main(args)


def test_emu3_gen(infer_backend):
    from swift.llm import infer_main, InferArguments
    args = InferArguments(
        model='BAAI/Emu3-Gen',
        infer_backend=infer_backend,
        stream=False,
        use_chat_template=False,
        top_k=2048,
        max_new_tokens=40960)
    infer_main(args)


if __name__ == '__main__':
    # test_emu3_gen('pt')
    test_deepseek_janus_pro_gene()
