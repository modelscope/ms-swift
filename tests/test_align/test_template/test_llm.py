import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _infer_model(pt_engine, system=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages += [{'role': 'user', 'content': '你好'}]
    resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    resp = pt_engine.infer([{
        'messages': messages,
        'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    }],
                           request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return messages


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


def test_llama():
    pt_engine = PtEngine('LLM-Research/Llama-3.2-3B-Instruct')
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


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template, get_model_tokenizer
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
    # test_llama()
    #
