# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Here is another way to register the model, by customizing the get_function.

The get_function just needs to return the model + tokenizer/processor.
"""

from typing import Any, Dict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from swift.llm import (InferRequest, Model, ModelGroup, ModelInfo, ModelMeta, PtEngine, RequestConfig, TemplateMeta,
                       register_model, register_template)

register_template(
    TemplateMeta(
        template_type='custom',
        prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
        prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
        chat_sep=['\n']))


def get_function(model_dir: str,
                 model_info: ModelInfo,
                 model_kwargs: Dict[str, Any],
                 load_model: bool = True,
                 **kwargs):
    # ref: https://github.com/modelscope/ms-swift/blob/main/swift/llm/model/register.py#L182
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=model_config, torch_dtype=model_info.torch_dtype, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
        ],
        template='custom',
        get_function=get_function,
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
