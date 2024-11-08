from swift.llm import TemplateInputs, get_model_tokenizer, get_template


def test_template():
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template_inputs = TemplateInputs([{
        'role': 'system',
        'content': 'AAA'
    }, {
        'role': 'user',
        'content': 'BBB'
    }, {
        'role': 'assistant',
        'content': 'CCC'
    }, {
        'role': 'user',
        'content': 'DDD'
    }])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(tokenizer.decode(inputs['input_ids']))


def test_mllm():
    _, tokenizer = get_model_tokenizer('qwen/Qwen2-VL-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template_inputs = TemplateInputs([{
        'role': 'system',
        'content': 'AAA'
    }, {
        'role': 'user',
        'content': '<image>BBB'
    }, {
        'role': 'assistant',
        'content': 'CCC'
    }, {
        'role': 'user',
        'content': 'DDD'
    }],
                                     images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'])
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(template.safe_tokenizer_decode(inputs['input_ids']))


if __name__ == '__main__':
    # test_template()
    test_mllm()
