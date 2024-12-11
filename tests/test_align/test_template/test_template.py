from swift.llm import TemplateInputs, get_model_tokenizer, get_template


def test_deepseek_v2_5():
    tokenizer = get_model_tokenizer('deepseek-ai/DeepSeek-V2.5-1210', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = TemplateInputs(messages=[{
        'role': 'user',
        'content': 'aaa'
    }, {
        'role': 'assistant',
        'content': 'bbb'
    }, {
        'role': 'user',
        'content': 'ccc'
    }])
    res = template.encode(inputs)
    template.print_inputs(res)
    template.template_backend = 'jinja'
    res = template.encode(inputs)
    template.print_inputs(res)


if __name__ == '__main__':
    test_deepseek_v2_5()
