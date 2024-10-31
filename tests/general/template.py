def test_template():
    from swift.llm import get_model_tokenizer, get_template, TemplateInputs
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template_inputs = TemplateInputs([
        {'role': 'system', 'content': 'AAA'},
        {'role': 'user', 'content': 'BBB'},
        {'role': 'assistant', 'content': 'CCC'},
        {'role': 'user', 'content': 'DDD'}])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(tokenizer.decode(inputs['input_ids']))


if __name__ == '__main__':
    test_template()
