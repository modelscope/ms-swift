from swift.llm import TemplateInputs, get_model_tokenizer, get_template


def test_deepseek_v2_5():
    tokenizer = get_model_tokenizer('deepseek-ai/DeepSeek-V2.5-1210', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = TemplateInputs({
        'messages': [{
            'role': 'system',
            'content': '000'
        }, {
            'role': 'user',
            'content': 'aaa'
        }, {
            'role': 'assistant',
            'content': 'bbb'
        }, {
            'role': 'user',
            'content': 'ccc'
        }]
    })
    res = template.encode(inputs)
    template.print_inputs(res)
    template.template_backend = 'jinja'
    res2 = template.encode(inputs)
    template.print_inputs(res2)
    assert res['input_ids'] == res2['input_ids']


def test_qwen2_5_math_reward():
    tokenizer = get_model_tokenizer('Qwen/Qwen2.5-Math-RM-72B', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = TemplateInputs({
        'messages': [{
            'role':
            'user',
            'content':
            'Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins '
            "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per "
            "fresh duck egg. How much in dollars does she make every day at the farmers' market?"
        }, {
            'role':
            'assistant',
            'content':
            "To determine how much Janet makes from selling the duck eggs at the farmers' market, we need to "
            'follow these steps:\n\n1. Calculate the total number of eggs laid by the ducks each day.\n2. '
            'Determine how many eggs Janet eats and bakes for herself each day.\n3. Find out how many eggs are '
            "left to be sold.\n4. Calculate the revenue from selling the remaining eggs at $2 per egg.\n\nLet's "
            "start with the first step:\n\n1. Janet's ducks lay 16 eggs per day.\n\nNext, we calculate how many "
            'eggs Janet eats and bakes for herself each day:\n\n2. Janet eats 3 eggs for breakfast every morning.'
            '\n3. Janet bakes 4 eggs for her friends every day.\n\nSo, the total number of eggs Janet eats and '
            'bakes for herself each day is:\n\\[ 3 + 4 = 7 \\text{ eggs} \\]\n\nNow, we find out how many eggs '
            'are left to be sold:\n\\[ 16 - 7 = 9 \\text{ eggs} \\]\n\nFinally, we calculate the revenue from '
            'selling the remaining eggs at $2 per egg:\n\\[ 9 \\times 2 = 18 \\text{ dollars} \\]\n\nTherefore, '
            "Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
        }]
    })
    res = template.encode(inputs)
    template.print_inputs(res)
    template.template_backend = 'jinja'
    res2 = template.encode(inputs)
    template.print_inputs(res)
    assert res['input_ids'] == res2['input_ids']
    assert len(res['input_ids']) == 364


def test_minimax():
    tokenizer = get_model_tokenizer('MiniMax/MiniMax-Text-01', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = TemplateInputs({
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant created by MiniMax based on MiniMax-Text-01 model.'
        }, {
            'role': 'user',
            'content': 'Hello!'
        }]
    })
    res = template.encode(inputs)
    template.print_inputs(res)
    assert tokenizer.decode(res['input_ids']) == (
        '<beginning_of_sentence>system ai_setting=assistant\nYou are a helpful assistant created by MiniMax based '
        'on MiniMax-Text-01 model.<end_of_sentence>\n<beginning_of_sentence>user name=user\nHello!<end_of_sentence>\n'
        '<beginning_of_sentence>ai name=assistant\n')


def test_minimax_vl():
    tokenizer = get_model_tokenizer('MiniMax/MiniMax-VL-01', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = TemplateInputs({
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant created by MiniMax based on MiniMax-VL-01 model.'
        }, {
            'role': 'user',
            'content': '<image>Describe this image.'
        }],
        'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    })
    res = template.encode(inputs)
    assert len(res['input_ids']) == 5877


def test_deepseek_v3_1():
    tokenizer = get_model_tokenizer('deepseek-ai/DeepSeek-V3.1', load_model=False)[1]
    template = get_template(tokenizer.model_meta.template, tokenizer)
    inputs = {
        'messages': [{
            'role': 'system',
            'content': '000'
        }, {
            'role': 'user',
            'content': 'aaa'
        }, {
            'role': 'assistant',
            'content': 'bbb'
        }, {
            'role': 'user',
            'content': 'ccc'
        }]
    }
    res = template.encode(inputs)
    template.print_inputs(res)
    template.template_backend = 'jinja'
    res2 = template.encode(inputs)
    template.print_inputs(res2)
    assert res['input_ids'] == res2['input_ids']


if __name__ == '__main__':
    # test_deepseek_v2_5()
    # test_qwen2_5_math_reward()
    # test_minimax()
    # test_minimax_vl()
    test_deepseek_v3_1()
