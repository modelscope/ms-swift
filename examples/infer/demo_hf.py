def infer_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from modelscope import snapshot_download
    model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct')
    adapter_dir = snapshot_download('swift/test_lora')
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map='auto')
    model = PeftModel.from_pretrained(model, adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }, {
        'role': 'user',
        'content': 'who are you?'
    }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'response: {response}')
    return response


def infer_swift():
    from swift.llm import get_model_tokenizer, get_template, InferRequest, RequestConfig, PtEngine
    from modelscope import snapshot_download
    from swift.tuners import Swift
    model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct')
    adapter_dir = snapshot_download('swift/test_lora')
    model, tokenizer = get_model_tokenizer(model_dir, device_map='auto')
    model = Swift.from_pretrained(model, adapter_dir)
    template = get_template(model.model_meta.template, tokenizer)
    engine = PtEngine.from_model_template(model, template)

    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant.'
    }, {
        'role': 'user',
        'content': 'who are you?'
    }]
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer([InferRequest(messages=messages)], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    return response


if __name__ == '__main__':
    response = infer_hf()
    response2 = infer_swift()
    assert response == response2
