# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Here is another way to register the model, by customizing the get_function.

The get_function just needs to return the model + tokenizer/processor.
"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import TemplateMeta, register_template
from swift.utils import Processor

register_template(
    TemplateMeta(
        template_type='custom',
        prefix=['<extra_id_0>System\n{{SYSTEM}}\n'],
        prompt=['<extra_id_1>User\n{{QUERY}}\n<extra_id_1>Assistant\n'],
        chat_sep=['\n']))


class MyModelLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        return AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=self.torch_dtype, trust_remote_code=True, **model_kwargs)


register_model(
    ModelMeta(
        model_type='custom',
        model_groups=[
            ModelGroup([Model('AI-ModelScope/Nemotron-Mini-4B-Instruct', 'nvidia/Nemotron-Mini-4B-Instruct')])
        ],
        loader=MyModelLoader,
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
