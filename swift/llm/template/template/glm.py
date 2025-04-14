# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_batch, load_video_cogvlm2


@dataclass
class GLMTemplateMeta(TemplateMeta):
    auto_add_bos: bool = True


class GLM4Z1Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        if not self.is_training:
            for message in inputs.messages:
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    message['content'] = message['content'].split('</think>')[-1].strip()
        return super()._swift_encode(inputs)


register_template(
    GLMTemplateMeta(
        LLMTemplateType.chatglm2,
        prefix=['{{SYSTEM}}'],
        prompt=['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'],
        chat_sep=['\n\n']))


@dataclass
class GLM4TemplateMeta(GLMTemplateMeta):
    prefix: Prompt = field(default_factory=list)
    prompt: Prompt = field(default_factory=lambda: ['<|user|>\n{{QUERY}}<|assistant|>\n'])
    chat_sep: Optional[Prompt] = field(default_factory=list)
    suffix: Prompt = field(default_factory=lambda: ['<|user|>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<|system|>\n{{SYSTEM}}'])

    default_tools_prompt: str = 'glm4'
    tool_prompt: Optional[Prompt] = field(default_factory=lambda: ['<|observation|>\n{{QUERY}}<|assistant|>\n'])
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>', '<|user|>', '<|observation|>'])


@dataclass
class GLM4Z1TemplateMeta(GLM4TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['[gMASK]<sop>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['[gMASK]<sop><|system|>\n{{SYSTEM}}'])


class GLM4VTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = inputs.images[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.processor.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            messages = inputs.messages
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.processor.apply_chat_template(messages, return_dict=True)
            encoded['images'] = inputs2['images']
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['position_ids'] = list(range(len(input_ids)))
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


# not '<|assistant|>\n'
register_template(GLM4TemplateMeta(MLLMTemplateType.glm4v, template_cls=GLM4VTemplate, suffix=['<|endoftext|>']))

register_template(GLM4TemplateMeta(LLMTemplateType.glm4))

register_template(GLM4Z1TemplateMeta(LLMTemplateType.glm4_z1, template_cls=GLM4Z1Template))

glm4z1rumination_system = (
    '你是一个专业的深度研究助手，通过提供的工具与模拟浏览器交互，来帮助用户完成深度信息调研和报告撰写任务。'
    '今年是 2025 年。\n\n'
    '<核心要求>\n'
    '- 首先分解用户请求，得到包含多个子要求的列表\n'
    '- 制定初始研究计划\n'
    '- 进行多轮迭代搜索和页面浏览（at least 10 function calls）：\n'
    '    * 根据已获得的信息调整研究计划和关键词\n'
    '    * 打开页面阅读，从发现的内容中识别新的关键概念/名词\n'
    '    * 从搜索结果中提取新的关键词继续搜索\n'
    '    * 访问并仔细阅读相关页面，识别新的关键概念/名词\n\n'
    '<重要配置>\n'
    '- 采用语言\n'
    '    * 搜索关键词：英语\n'
    '    * 思考：英语\n\n'
    '<可调用的工具列表>\n\n'
    '[{"name": "search", "description": "Execute a search query and return search results. '
    'Use this function when you need to find information about a specific topic.", '
    '"parameters": {"type": "object", "properties": {"query": {"type": "string", '
    '"description": "Search query string, use English words unless it is a proper name in Chinese"}}, '
    '"required": ["query"], "additionalProperties": false}}, '
    '{"name": "click", "description": "Click a link in the search results and navigate to the corresponding page. '
    'Use this function when you need to view detailed content of a specific search result.", '
    '"parameters": {"type": "object", "properties": {"link_id": {"type": "integer", '
    '"description": "The link ID to click (from the sequence number in search results)"}}, '
    '"required": ["link_id"], "additionalProperties": false}}, '
    '{"name": "open", "description": "Open a specific website. Get content from any website with its URL.", '
    '"parameters": {"type": "object", "properties": {"url": {"type": "string", '
    '"description": "The target website URL or domain"}}, "required": ["url"], "additionalProperties": false}}, '
    '{"name": "finish", "description": "Finish the task. '
    'Use this function when you have found the information you need.", '
    '"parameters": {"type": "object", "properties": {}, "additionalProperties": false}}]')

register_template(
    GLM4Z1TemplateMeta(
        LLMTemplateType.glm4_z1_rumination, template_cls=GLM4Z1Template, default_system=glm4z1rumination_system))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(GLM4TemplateMeta(LLMTemplateType.codegeex4, default_system=codegeex4_system))

register_template(
    TemplateMeta(
        LLMTemplateType.longwriter_llama, ['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'],
        system_prefix=['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))


class CogTemplate(Template):
    placeholder_tokens = ['<|reserved_special_token_0|>']

    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        return []

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        model = self.model
        image = inputs.images or []
        history_inputs = inputs.to_history()
        inputs2 = model.build_conversation_input_ids(
            self.processor, query=history_inputs['query'], history=history_inputs['history'], images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        encoded['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        encoded['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            encoded['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            encoded['images'] = [[img.to(dtype=self.model_info.torch_dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                encoded['cross_images'] = [[cross_img.to(dtype=self.model_info.torch_dtype)]
                                           for cross_img in inputs2['cross_images']]
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.cogagent_chat,
        prefix=['<s>'],
        prompt=[' [INST] {{QUERY}} [/INST] '],
        chat_sep=[],
        suffix=['</s>'],
        template_cls=CogTemplate,
    ))

register_template(
    TemplateMeta(
        MLLMTemplateType.cogagent_vqa,
        prefix=['<s>'],
        prompt=['<EOI>Question: {{QUERY}} Answer:'],
        chat_sep=None,
        suffix=['</s>'],
        template_cls=CogTemplate))


@dataclass
class CogVLMTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [['bos_token_id']])
    prompt: Prompt = field(default_factory=lambda: ['Question: {{QUERY}} Answer:'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['\n'])


register_template(CogVLMTemplateMeta(MLLMTemplateType.cogvlm, template_cls=CogTemplate))

register_template(CogVLMTemplateMeta(MLLMTemplateType.cogvlm2, template_cls=CogTemplate))


class Cog2VideoTemplate(CogTemplate):
    use_model = True

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        model = self.model
        encoded = super(CogTemplate, self)._encode(inputs)
        videos_path = inputs.videos or []
        video = load_batch(videos_path, load_video_cogvlm2)
        history_inputs = inputs.to_history()
        inputs2 = model.build_conversation_input_ids(
            self.processor,
            query=history_inputs['query'],
            history=history_inputs['history'],
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        encoded['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        encoded['input_ids'] = input_ids[:1] + [self.processor.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            encoded['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            encoded['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return encoded


register_template(CogVLMTemplateMeta(
    MLLMTemplateType.cogvlm2_video,
    template_cls=Cog2VideoTemplate,
))


class GLMEdgeVTemplate(Template):
    placeholder_tokens = ['<|begin_of_image|>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<|begin_of_image|>' * 578]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if images:
            encoded['pixel_values'] = torch.tensor(self.processor(images).pixel_values)
        return encoded


register_template(
    GLM4TemplateMeta(
        MLLMTemplateType.glm_edge_v,
        prompt=['<|user|>\\n{{QUERY}}\\n<|assistant|>\\n'],
        chat_sep=['\\n'],
        system_prefix=['<|system|>\\n{{SYSTEM}}\\n'],
        suffix=['<|endoftext|>'],
        template_cls=GLMEdgeVTemplate,
    ))
