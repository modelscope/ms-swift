def test_vllm():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import (ModelType, get_vllm_engine, get_default_template_type, get_template, inference_vllm,
                           inference_stream_vllm)

    model_type = ModelType.qwen_7b_chat
    llm_engine = get_vllm_engine(model_type)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    llm_engine.generation_config.max_new_tokens = 256
    generation_info = {}

    request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
    resp_list = inference_vllm(llm_engine, template, request_list, generation_info=generation_info)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")
    print(generation_info)

    # stream
    history1 = resp_list[1]['history']
    request_list = [{'query': '这有什么好吃的', 'history': history1}]
    gen = inference_stream_vllm(llm_engine, template, request_list, generation_info=generation_info)
    query = request_list[0]['query']
    print_idx = 0
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        resp = resp_list[0]
        response = resp['response']
        delta = response[print_idx:]
        print(delta, end='', flush=True)
        print_idx = len(response)
    print()

    history = resp_list[0]['history']
    print(f'history: {history}')
    print(generation_info)

    # batched
    n_batched = 100
    request_list = [{'query': '晚上睡不着觉怎么办?'} for i in range(n_batched)]
    resp_list = inference_vllm(llm_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
    assert len(resp_list) == n_batched
    print(resp_list[0]['history'])
    print(generation_info)

    request_list = [{'query': '晚上睡不着觉怎么办?'} for i in range(n_batched)]
    gen = inference_stream_vllm(llm_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
    for resp_list in gen:
        pass
    assert len(resp_list) == n_batched
    print(resp_list[0]['history'])
    print(generation_info)


if __name__ == '__main__':
    test_vllm()
