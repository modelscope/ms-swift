import os
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_multilora(infer_request: 'InferRequest', infer_backend: Literal['vllm', 'pt']):
    # Dynamic LoRA
    adapter_path = snapshot_download('swift/test_lora')
    args = BaseArguments.from_pretrained(adapter_path)
    if infer_backend == 'pt':
        engine = PtEngine('Qwen/Qwen2.5-7B-Instruct')
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine('Qwen/Qwen2.5-7B-Instruct', enable_lora=True, max_loras=1, max_lora_rank=16)
    template = get_template(args.template, engine.processor, args.system)
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_request = AdapterRequest('lora1', adapter_path)

    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template, adapter_request=adapter_request)
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')
    # origin model
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template, adapter_request=adapter_request)
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')


def infer_pt(infer_request: 'InferRequest'):
    adapter_path = snapshot_download('swift/test_lora')
    args = BaseArguments.from_pretrained(adapter_path)
    model, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-7B-Instruct')
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # use lora
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')


if __name__ == '__main__':
    from swift.llm import (PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest,
                           get_model_tokenizer)
    from swift.tuners import Swift
    from modelscope import snapshot_download
    infer_request = InferRequest(messages=[{'role': 'user', 'content': '你是谁'}])
    infer_multilora(infer_request, 'vllm')
    # infer_pt(infer_request)