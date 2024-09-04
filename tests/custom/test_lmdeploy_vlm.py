def test_lmdeploy_vlm():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    from swift.llm import (ModelType, get_lmdeploy_engine, get_default_template_type, get_template, inference_lmdeploy,
                           inference_stream_lmdeploy)

    model_type = ModelType.deepseek_vl_1_3b_chat
    lmdeploy_engine = get_lmdeploy_engine(model_type)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    lmdeploy_engine.generation_config.max_new_tokens = 256
    lmdeploy_engine.generation_config.logprobs = 2
    generation_info = {}

    request_list = [{
        'query':
        '这两张图片有什么区别：'
        '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>'
        '<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>'
    }, {
        'query': '你好'
    }]
    resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")
        print(f"len(logprobs): {len(resp['logprobs'])}")
    print(generation_info)

    # stream
    history0 = resp_list[0]['history']
    request_list = [{'query': '有几只羊', 'history': history0}]
    gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
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
    print(f"len(logprobs): {len(resp_list[0]['logprobs'])}")

    # batched
    n_batched = 1000
    request_list = [{
        'query':
        '这两张图片有什么区别：'
        '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>'
        '<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>'
    } for i in range(n_batched)]
    resp_list = inference_lmdeploy(
        lmdeploy_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
    assert len(resp_list) == n_batched
    print(resp_list[0]['history'])
    print(generation_info)

    request_list = [{
        'query':
        '这两张图片有什么区别：'
        '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>'
        '<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>'
    } for i in range(n_batched)]
    gen = inference_stream_lmdeploy(
        lmdeploy_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
    for resp_list in gen:
        pass
    assert len(resp_list) == n_batched
    print(resp_list[0]['history'])
    print(generation_info)


if __name__ == '__main__':
    test_lmdeploy_vlm()
