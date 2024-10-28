def test_template():
    from swift.llm import get_model_tokenizer, get_template, get_model_meta, TemplateInputs
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', load_model=False)
    model_meta = get_model_meta(tokenizer.model_info.model_type)
    template = get_template(model_meta.template, tokenizer)
    template_inputs = TemplateInputs([{'role': 'user', 'content': 'hello! who are you'}])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(tokenizer.decode(inputs['input_ids']))


if __name__ == '__main__':
    test_template()
