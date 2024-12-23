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
    pt_engine = PtEngine('TeleAI/TeleChat2-7B', torch_dtype=torch.float16)
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '你是谁？'}])
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '你是谁？'}])


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
    # test_telechat()
    # test_glm_edge()
    # test_llama()
    # test_openbuddy()
    test_megrez()
