def test_template():
    from swift.llm import get_model_tokenizer, get_template, get_model_meta
    model, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', load_model=False)
    model_meta = get_model_meta(model.model_info.model_type)
    template = get_template(model_meta.template)
    inputs_request = InferRequest([{'role': 'user', 'content': 'hello! who are you'}])
