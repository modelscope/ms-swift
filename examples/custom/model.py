# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.template import TemplateMeta, register_template

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
        ignore_patterns=['nemo'],
        is_multimodal=False,
    ))

if __name__ == '__main__':
    infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
    request_config = RequestConfig(max_tokens=512, temperature=0)
    engine = TransformersEngine('AI-ModelScope/Nemotron-Mini-4B-Instruct')
    response = engine.infer([infer_request], request_config)
    swift_response = response[0].choices[0].message.content

    engine.template.template_backend = 'jinja'
    response = engine.infer([infer_request], request_config)
    jinja_response = response[0].choices[0].message.content
    assert swift_response == jinja_response, f'swift_response: {swift_response}\njinja_response: {jinja_response}'
    print(f'response: {swift_response}')
