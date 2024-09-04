def test_pt_vlm():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import (ModelType, get_model_tokenizer, get_default_template_type, get_template, inference,
                           inference_stream)

    model_type = ModelType.internvl2_2b
    model, tokenizer = get_model_tokenizer(model_type)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    model.generation_config.max_new_tokens = 256
    model.generation_config.return_dict_in_generate = True
    generation_info = {}
    query = 'who are you?'
    resp = inference(model, template, query=query, generation_info=generation_info)
    print(f'query: {query}')
    print(f"response: {resp['response']}")
    print(generation_info)
    print(resp.keys())

    # stream
    images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
    history = resp['history']
    query = '有几只羊'
    gen = inference_stream(
        model, template, query=query, history=history, images=images, generation_info=generation_info)
    print_idx = 0
    print(f'query: {query}\nresponse: ', end='')
    for resp in gen:
        response = resp['response']
        delta = response[print_idx:]
        print(delta, end='', flush=True)
        print_idx = len(response)
    print()

    history = resp['history']
    print(f'history: {history}')
    print(generation_info)
    print(resp.keys())


if __name__ == '__main__':
    test_pt_vlm()
