import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['SWIFT_DEBUG'] = '1'


def _infer_model(pt_engine, system=None, messages=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    else:
        messages = messages.copy()
    resp = pt_engine.infer([{
        'messages': messages,
    }], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen2_5():
    pt_engine = PtEngine('Qwen/Qwen2.5-3B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qwen1half():
    pt_engine = PtEngine('Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_glm4():
    # The Jinja prompt is missing \n.
    pt_engine = PtEngine('ZhipuAI/glm-4-9b-chat')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qwq():
    pt_engine = PtEngine('Qwen/QwQ-32B-Preview')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_internlm():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-chat-7b')
    _infer_model(pt_engine)


def test_internlm2():
    # pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm2-1_8b')
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm2_5-1_8b-chat')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_yi_coder():
    pt_engine = PtEngine('01ai/Yi-Coder-1.5B-Chat')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_yi():
    pt_engine = PtEngine('01ai/Yi-6B-Chat')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_deepseek_moe():
    pt_engine = PtEngine('deepseek-ai/deepseek-moe-16b-chat')
    _infer_model(pt_engine)


def test_codegeex4():
    # jinja is missing a prefix.
    pt_engine = PtEngine('ZhipuAI/codegeex4-all-9b')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_telechat():
    pt_engine = PtEngine('TeleAI/TeleChat-12B', torch_dtype=torch.float16)
    messages = [{'role': 'user', 'content': '你是谁'}]
    response = _infer_model(pt_engine, messages=messages)
    assert response == ('我是中国电信星辰语义大模型，英文名TeleChat，是由中国电信自主研发的生成式大语言模型。\n\n'
                        '我基于Transformer-decoder结构，学习了海量知识，包括百科、书籍、论坛、党政媒体、GitHub代码、专业领域知识等，'
                        '具备自然语言处理、语义理解、内容创作和逻辑推理等能力，可以与人类进行对话互动和情感交流，还能提供知识问答、创作写作、'
                        '代码生成等服务，希望能为人类带来更加智能、高效和便捷的工作与生活体验。')


def test_telechat2():
    pt_engine = PtEngine('TeleAI/TeleChat2-7B', torch_dtype=torch.float16)
    messages = [{'role': 'system', 'content': '你是一个乐于助人的智能助手，请使用用户提问的语言进行有帮助的问答'}, {'role': 'user', 'content': '你好'}]
    response = _infer_model(pt_engine, messages=messages)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages)
    assert response == response2


def test_glm_edge():
    pt_engine = PtEngine('ZhipuAI/glm-edge-1.5b-chat')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_llama():
    # pt_engine = PtEngine('LLM-Research/Meta-Llama-3.1-8B-Instruct-BNB-NF4')
    # pt_engine = PtEngine('LLM-Research/Meta-Llama-3.1-8B-Instruct')
    # pt_engine = PtEngine('LLM-Research/Meta-Llama-3-8B-Instruct')
    pt_engine = VllmEngine('LLM-Research/Llama-3.2-1B-Instruct')
    # pt_engine = PtEngine('AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF')
    # pt_engine = PtEngine('unsloth/Llama-3.3-70B-Instruct-bnb-4bit')

    res = _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    res2 = _infer_model(pt_engine, system='')
    assert res == res2, f'res: {res}, res2: {res2}'


def test_openbuddy():
    # pt_engine = PtEngine('OpenBuddy/openbuddy-yi1.5-34b-v21.3-32k')
    pt_engine = PtEngine('OpenBuddy/openbuddy-nemotron-70b-v23.2-131k')
    # pt_engine = PtEngine('OpenBuddy/openbuddy-llama3.3-70b-v24.3-131k')
    res = _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    res2 = _infer_model(pt_engine)
    assert res == res2, f'res: {res}, res2: {res2}'


def test_megrez():
    pt_engine = PtEngine('InfiniAI/Megrez-3b-Instruct')
    res = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    res2 = _infer_model(pt_engine)
    assert res == res2, f'res: {res}, res2: {res2}'


def test_skywork_o1():
    pt_engine = PtEngine('AI-ModelScope/Skywork-o1-Open-Llama-3.1-8B')
    res = _infer_model(
        pt_engine,
        messages=[{
            'role':
            'user',
            'content':
            ('Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits '
             'all her apples equally among herself and her 2 siblings. How many apples does each person get?')
        }])
    assert res == ("To solve the problem, let's break it down into a series of logical steps:\n\n1. **Initial Number "
                   'of Apples**: Jane starts with 12 apples.\n2. **Apples Given Away**: Jane gives 4 apples to her '
                   'friend Mark. So, the number of apples she has now is:\n   \\[\n   12 - 4 = 8\n   \\]\n3. **Apples '
                   'Bought**: Jane then buys 1 more apple. So, the number of apples she has now is:\n   \\[\n   '
                   '8 + 1 = 9\n   \\]\n4. **Apples Split Equally')


def test_internlm2_reward():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm2-1_8b-reward')
    messages = [{
        'role': 'user',
        'content': "Hello! What's your name?"
    }, {
        'role': 'assistant',
        'content': 'My name is InternLM2! A helpful AI assistant. What can I do for you?'
    }]
    res = _infer_model(pt_engine, messages=messages)
    pt_engine.default_template.template_backend = 'jinja'
    res2 = _infer_model(pt_engine, messages=messages)
    assert res == res2 == '0.48681640625'


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template, get_model_tokenizer, VllmEngine
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen2_5()
    # test_qwen1half()
    # test_qwq()
    # test_internlm()
    # test_internlm2()
    # test_yi_coder()
    # test_yi()
    # test_deepseek_moe()
    # test_codegeex4()
    # test_glm4()
    test_telechat()
    # test_telechat2()
    # test_glm_edge()
    # test_llama()
    # test_openbuddy()
    # test_megrez()
    # test_skywork_o1()
    # test_internlm2_reward()
