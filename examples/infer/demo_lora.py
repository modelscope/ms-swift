import os
from typing import Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_multilora(infer_request: 'InferRequest', infer_backend: Literal['vllm', 'pt']):
    # Dynamic LoRA
    adapter_path = safe_snapshot_download('swift/test_lora')
    adapter_path2 = safe_snapshot_download('swift/test_lora2')
    args = BaseArguments.from_pretrained(adapter_path)
    if infer_backend == 'pt':
        engine = TransformersEngine(args.model)
    elif infer_backend == 'vllm':
        from swift.infer_engine import VllmEngine
        engine = VllmEngine(args.model, enable_lora=True, max_loras=1, max_lora_rank=16)
    template = get_template(engine.processor, template_type=args.template, default_system=args.system)
    engine.template = template
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_request = AdapterRequest('lora1', adapter_path)
    adapter_request2 = AdapterRequest('lora2', adapter_path2)

    # use lora
    resp_list = engine.infer([infer_request], request_config, adapter_request=adapter_request)
    response = resp_list[0].choices[0].message.content
    print(f'lora1-response: {response}')
    # origin model
    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # use lora
    resp_list = engine.infer([infer_request], request_config, adapter_request=adapter_request2)
    response = resp_list[0].choices[0].message.content
    print(f'lora2-response: {response}')


def infer_lora(infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_path = safe_snapshot_download('swift/test_lora')
    args = BaseArguments.from_pretrained(adapter_path)
    # method1
    # engine = TransformersEngine(args.model, adapters=[adapter_path])
    # template = get_template(engine.processor, args.system, template_type=args.template)
    # engine.template = template

    # method2
    # model, processor = args.get_model_processor()
    # model = Swift.from_pretrained(model, adapter_path)
    # template = args.get_template(processor)
    # engine = TransformersEngine(model, template=template)

    # method3
    model, tokenizer = get_model_processor(args.model)
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(tokenizer, args.system, template_type=args.template)
    engine = TransformersEngine(model, template=template)

    resp_list = engine.infer([infer_request], request_config)
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')


if __name__ == '__main__':
    from swift import (TransformersEngine, RequestConfig, AdapterRequest, InferRequest, BaseArguments,
                       get_model_processor, safe_snapshot_download, Swift, get_template)
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
    # infer_lora(infer_request)
    infer_multilora(infer_request, 'pt')
