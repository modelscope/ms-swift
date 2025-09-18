# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from swift.llm import get_packed_seq_params
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, Word, findall
from ..vision_utils import load_batch, load_video_cogvlm2, load_video_hf
from .utils import ThinkingTemplate


@dataclass
class GLMTemplateMeta(TemplateMeta):
    auto_add_bos: bool = True


class GLM4Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        res_context_list, loss_scale_list, answer_len = super()._swift_encode(inputs)
        for i, res_context in enumerate(res_context_list):
            # The last round or is tool_call.
            if isinstance(res_context, str) and res_context.endswith('<|assistant|>\n') and (
                    i + 1 >= len(res_context_list) or '<|observation|>' in res_context_list[i + 1]):
                res_context_list[i] = res_context_list[i][:-len('\n')]
        return res_context_list, loss_scale_list, answer_len

    def decode(self, *args, **kwargs):
        response = super().decode(*args, **kwargs)
        return response.lstrip('\n')


class GLM4_0414Template(ThinkingTemplate, GLM4Template):
    pass


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

    agent_template: str = 'glm4'
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>', '<|user|>', '<|observation|>'])


@dataclass
class GLM4_0414TemplateMeta(GLM4TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['[gMASK]<sop>'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['[gMASK]<sop><|system|>\n{{SYSTEM}}'])
    agent_template: str = 'glm4_0414'


@dataclass
class GLM4_5TemplateMeta(GLM4_0414TemplateMeta):
    agent_template: str = 'glm4_5'


class GLM4_1VTemplateMeta(GLM4_0414TemplateMeta):
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['[gMASK]<sop><|system|>{{SYSTEM}}'])


class GLM4VTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            return ['<|begin_of_image|><|endoftext|><|end_of_image|>']
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


class GLM4_1VTemplate(Template):
    begin_of_image_token = 151339
    end_of_image_token = 151340
    image_token = 151343
    begin_of_video_token = 151341
    end_of_video_token = 151342
    video_token = 151344

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        # TODO: model video infer bug
        if self.mode == 'vllm':
            if media_type == 'image':
                return ['<|begin_of_image|><|image|><|end_of_image|>']
            elif media_type == 'video':
                return ['<|begin_of_video|><|video|><|end_of_video|>']
        assert media_type in ['image']
        if media_type == 'image':
            return [[-100]]
        elif media_type == 'video':
            return [[-200]]

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        image_idx_list = findall(input_ids, -100)
        video_idx_list = findall(input_ids, -200)
        if image_idx_list:
            images = inputs.images
            image_inputs = processor.image_processor(images=images, return_tensors='pt')
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['image_grid_thw'] = image_grid_thw = image_inputs['image_grid_thw']
            merge_length = processor.image_processor.merge_size**2
            added_tokens_len = 0
            for i, idx in enumerate(image_idx_list):
                num_image_tokens = image_grid_thw[i].prod() // merge_length
                image_tokens = [self.begin_of_image_token
                                ] + [self.image_token] * num_image_tokens + [self.end_of_image_token]

                input_ids = input_ids[:added_tokens_len + idx] + image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * len(image_tokens) + labels[added_tokens_len
                                                                                                   + idx + 1:]
                added_tokens_len += len(image_tokens) - 1

        if video_idx_list:
            # TODO: model video infer bug
            assert len(
                video_idx_list) <= 1, f'GLM4.1V model only support 1 video, but detected {len(video_idx_list)} <video> '
            assert not image_idx_list, "GLM4.1V model doesn't support inputs containing both video and images"

            video_fnames = inputs.videos
            from transformers.video_utils import load_video
            from transformers.image_utils import load_image
            import numpy as np
            video_metadata = []
            videos = []
            for fname in video_fnames:
                if isinstance(fname, (list, tuple)) and isinstance(fname[0], str):
                    video = [np.array(load_image(image_fname)) for image_fname in fname]
                    # create a 4D video because `load_video` always returns a 4D array
                    video = np.stack(video)
                    metadata = None
                else:
                    video, metadata = load_video(fname)
                videos.append(video)
                video_metadata.append(metadata)
            videos = [videos]
            video_metadata = [video_metadata]

            videos_inputs = processor.video_processor(videos=videos, video_metadata=video_metadata, return_tensors='pt')
            encoded['pixel_values_videos'] = videos_inputs['pixel_values_videos']
            encoded['video_grid_thw'] = video_grid_thw = videos_inputs['video_grid_thw']
            timestamps = videos_inputs.pop('timestamps')
            num_frames = len(video_grid_thw)
            video_structure = [self.begin_of_video_token]
            if hasattr(timestamps, 'tolist'):
                timestamps_list = timestamps.tolist()[0]
            else:
                timestamps_list = timestamps[0] if isinstance(timestamps[0], list) else timestamps
            unique_timestamps = []
            for idx in range(0, len(timestamps_list)):
                unique_timestamps.append(timestamps_list[idx])
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)
            merge_length = processor.video_processor.merge_size**2
            added_tokens_len = 0
            for frame_idx in range(num_frames):
                timestamp_sec = selected_timestamps[frame_idx]
                num_image_tokens = video_grid_thw[frame_idx].prod() // merge_length
                timestamp_sec_token = processor.tokenizer(str(timestamp_sec))['input_ids']
                frame_structure = [self.begin_of_image_token] + [self.image_token] * num_image_tokens + \
                    [self.end_of_image_token] + timestamp_sec_token
                video_structure += frame_structure
            video_structure += [self.end_of_video_token]

            for i, idx in enumerate(video_idx_list):
                # BUG in GLM4.1V?: All video placeholder take same tokens
                # https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/glm4v/processing_glm4v.py#L165-L194
                input_ids = input_ids[:added_tokens_len + idx] + video_structure + \
                    input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * len(video_structure) + \
                        labels[added_tokens_len + idx + 1:]
                added_tokens_len += len(video_structure) - 1

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['position_ids'] = list(range(len(input_ids)))
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: check video
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        inputs_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}


register_template(GLM4TemplateMeta(MLLMTemplateType.glm4v, template_cls=GLM4VTemplate, suffix=['<|endoftext|>']))

register_template(GLM4TemplateMeta(LLMTemplateType.glm4, template_cls=GLM4Template))

register_template(GLM4_0414TemplateMeta(LLMTemplateType.glm4_0414, template_cls=GLM4_0414Template))


class GLM4_5Template(ThinkingTemplate):
    no_think_prefix = '<think></think>\n'
    history_think_prefix = '<think></think>\n'

    def _jinja_encode(self, inputs: StdTemplateInputs):
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'],
                                                             str) and message['content'].endswith('<|observation|>'):
                message['content'] = message['content'][:-len('<|observation|>')]
        return super()._jinja_encode(inputs)


register_template(GLM4_5TemplateMeta(LLMTemplateType.glm4_5, template_cls=GLM4_5Template))

register_template(GLM4_1VTemplateMeta(MLLMTemplateType.glm4_1v, template_cls=GLM4_1VTemplate))


class GLM4_5VTemplate(GLM4_5Template):
    placeholder_tokens = ['<|image|>']
    support_padding_free = True  # https://github.com/huggingface/transformers/issues/39685
    use_model = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|begin_of_image|><|image|><|end_of_image|>']
        elif media_type == 'video':
            return ['<|begin_of_video|><|video|><|end_of_video|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        for mm_type in ['image', 'video']:
            mm_token = f'<|{mm_type}|>'
            mm_token_id = self._tokenize(mm_token)[0]

            idx_list = findall(input_ids, mm_token_id)
            if idx_list:
                split_token = self._tokenize('\n')[0]
                mm_data = getattr(inputs, f'{mm_type}s')
                if mm_type == 'image':
                    kwargs = {'images': mm_data}
                else:
                    videos, video_metadata = load_video_hf(mm_data)
                    kwargs = {'videos': [videos], 'video_metadata': [video_metadata]}
                mm_inputs = self.processor(text='\n'.join([mm_token] * len(mm_data)), return_tensors='pt', **kwargs)
                splited_tokens = self._split_list(mm_inputs['input_ids'][0].tolist(), split_token)
                for key in ['input_ids', 'token_type_ids', 'attention_mask']:
                    mm_inputs.pop(key, None)
                input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                    input_ids, encoded['labels'], encoded['loss_scale'], idx_list, lambda i: splited_tokens[i])
                encoded.update(mm_inputs)
        encoded['input_ids'] = input_ids
        return encoded

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        base_model = self.get_base_model(self.model)
        position_ids, _ = base_model.model.get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=inputs.get('attention_mask'))
        return self._concat_text_position_ids(position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            # https://github.com/huggingface/transformers/pull/40194
            if text_position_ids.shape[0] == 1:
                res.update(get_packed_seq_params(text_position_ids))
        return res


register_template(GLM4_0414TemplateMeta(MLLMTemplateType.glm4_5v, template_cls=GLM4_5VTemplate))

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
    GLM4_0414TemplateMeta(
        LLMTemplateType.glm4_z1_rumination, template_cls=GLM4_0414Template, default_system=glm4z1rumination_system))

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
