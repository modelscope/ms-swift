import os
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_multilora(infer_request: 'InferRequest', infer_backend: Literal['vllm', 'pt']):
    adapter_path = snapshot_download('swift/test_lora')
    args = BaseArguments.from_pretrained(adapter_path)
    if infer_backend == 'pt':
        engine = PtEngine('Qwen/Qwen2.5-7B-Instruct')
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine('Qwen/Qwen2.5-7B-Instruct', enable_lora=True, max_loras=1, max_lora_rank=16)
    template = get_template(args.template, engine.processor, args.system)
    request_config = RequestConfig(max_tokens=512, temperature=0)
    lora_request = AdapterRequest('lora1', adapter_path)

    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template, adapter_request=lora_request)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # origin model
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # use lora
    resp_list = engine.infer([infer_request], request_config, template=template, adapter_request=lora_request)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')


def infer_pt(infer_request: 'InferRequest'):
    adapter_path = snapshot_download('swift/test_lora')
    args = BaseArguments.from_pretrained(adapter_path)
    engine = PtEngine('Qwen/Qwen2.5-7B-Instruct')
    template = get_template(args.template, engine.processor, args.system)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # use lora
    engine.add_adapter(adapter_path)
    resp_list = engine.infer([infer_request], request_config, template=template)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, AdapterRequest, get_template, BaseArguments, InferRequest
    from modelscope import snapshot_download
    infer_request = InferRequest(messages=[{'role': 'user', 'content': '你是谁'}])
    infer_multilora(infer_request, 'vllm')
    # infer_pt(infer_request)
