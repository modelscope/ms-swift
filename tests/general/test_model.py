import os

import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def test_qwen2():
    import os
    from swift.llm import get_model_tokenizer
    model, tokenizer = get_model_tokenizer('Qwen/Qwen2-7B-Instruct', load_model=False)
    print(f'model: {model}, tokenizer: {tokenizer}')
    # test hf
    model, tokenizer = get_model_tokenizer('Qwen/Qwen2-7B-Instruct', load_model=False, use_hf=True)

    model, tokenizer = get_model_tokenizer(
        'Qwen/Qwen2-7B-Instruct', torch.float32, device_map='cuda:0', attn_impl='flash_attn')
    print(f'model: {model}, tokenizer: {tokenizer}')


def test_modelscope_hub():
    from swift.llm import get_model_tokenizer
    model, tokenizer = get_model_tokenizer('Qwen/Qwen2___5-Math-1___5B-Instruct/', load_model=False)


if __name__ == '__main__':
    test_qwen2()
    # test_modelscope_hub()
