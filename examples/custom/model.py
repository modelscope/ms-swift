# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import (InferRequest, Model, ModelGroup, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
                       get_model_tokenizer_with_flash_attn, register_model, register_template)

register_template(
    TemplateMeta(
        template_type='custom',
        prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
        prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
        chat_sep=['\n']))

register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
        ],
        template='custom',
        get_function=get_model_tokenizer_with_flash_attn,
        ignore_patterns=['nemo'],
        is_multimodal=False,
    ))

if __name__ == '__main__':
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
    request_config = RequestConfig(max_tokens=512, temperature=0)
    engine = PtEngine('AI-ModelScope/Nemotron-Mini-4B-Instruct')
    response = engine.infer([infer_request], request_config)
    swift_response = response[0].choices[0].message.content

    engine.default_template.template_backend = 'jinja'
    response = engine.infer([infer_request], request_config)
    jinja_response = response[0].choices[0].message.content
    assert swift_response == jinja_response, f'swift_response: {swift_response}\njinja_response: {jinja_response}'
    print(f'response: {swift_response}')
