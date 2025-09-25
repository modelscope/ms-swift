# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
import transformers
from packaging import version
from PIL import Image
from torch import nn
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.llm import get_packed_seq_params, to_float_dtype
from swift.utils import get_env_args, is_deepspeed_enabled
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import register_template
from ..template_inputs import StdTemplateInputs
from ..template_meta import TemplateMeta
from ..utils import Context, Word, findall
from ..vision_utils import load_audio, load_batch, load_video_ovis2, load_video_ovis2_5
from .llama import Llama3TemplateMeta
from .utils import DEFAULT_SYSTEM, ChatmlTemplateMeta, ThinkingTemplate


@dataclass
class QwenTemplateMeta(ChatmlTemplateMeta):
    default_system: Optional[str] = DEFAULT_SYSTEM
    auto_add_bos: bool = False
    stop_words: List[Word] = field(default_factory=lambda: ['<|endoftext|>'])
    agent_template: str = 'hermes'


@dataclass
class Qwen2_5TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


@dataclass
class Qwen2_5MathTemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = 'Please reason step by step, and put your final answer within \\boxed{}.'


qwq_preview_system = ('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                      'You should think step-by-step.')

register_template(QwenTemplateMeta(LLMTemplateType.qwen))
register_template(Qwen2_5TemplateMeta(LLMTemplateType.qwen2_5))
register_template(QwenTemplateMeta(LLMTemplateType.qwq_preview, default_system=qwq_preview_system))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwq, default_system=None, response_prefix='<think>\n', template_cls=ThinkingTemplate))


class Qwen3Template(ThinkingTemplate):
    no_think_prefix = '<think>\n\n</think>\n\n'


register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=Qwen3Template))

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_thinking, default_system=None, response_prefix='<think>\n',
        template_cls=ThinkingTemplate))

register_template(QwenTemplateMeta(LLMTemplateType.qwen3_nothinking, default_system=None))

register_template(QwenTemplateMeta(LLMTemplateType.qwen3_coder, default_system=None, agent_template='qwen3_coder'))


class Qwen3RerankerTemplate(Template):
    instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    def _preprocess_inputs(self, inputs: StdTemplateInputs) -> None:
        if inputs.system is not None:
            instruction = inputs.system
            inputs.system = None
        else:
            instruction = self.instruction
        query = inputs.messages[0]['content']
        document = inputs.messages[1]['content']
        user_message = '<Instruct>: ' + instruction + '\n' + '<Query>: ' + query + '\n' + '<Document>: ' + document
        inputs.messages = [{'role': 'user', 'content': user_message}]
        return inputs

    def prepare_engine_kwargs(self) -> Dict[str, Any]:
        if self.mode == 'vllm':
            return {
                'hf_overrides': {
                    'architectures': ['Qwen3ForSequenceClassification'],
                    'classifier_from_token': ['no', 'yes'],
                    'is_original_qwen3_reranker': True,
                }
            }
        else:
            return super().prepare_engine_kwargs()


qwen3_reranker_system = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be \"yes\" or \"no\".')

register_template(
    QwenTemplateMeta(
        LLMTemplateType.qwen3_reranker,
        default_system=qwen3_reranker_system,
        response_prefix='<think>\n\n</think>\n\n',
        template_cls=Qwen3RerankerTemplate))

register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math))


class QwenPRMTemplate(Template):
    cot_process_placeholder = '<extra_0>'

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        super()._preprocess_inputs(inputs)
        total_content = '\n'.join([message['content'] or '' for message in inputs.messages])
        if self.cot_process_placeholder not in total_content:
            inputs.messages[-1]['content'] = inputs.messages[-1]['content'] + self.cot_process_placeholder

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        step_sep_id = self.tokenizer.encode(self.cot_process_placeholder)[0]
        token_masks = (input_ids == step_sep_id)
        return self.make_step_rewards(logits, token_masks)


register_template(Qwen2_5MathTemplateMeta(LLMTemplateType.qwen2_5_math_prm, template_cls=QwenPRMTemplate))


class QwenVLTemplate(Template):
    load_images = False

    @staticmethod
    def _load_image(image, load_images: bool):
        if not load_images and isinstance(image, str) and (image.startswith('data:') or len(image) > 200):
            load_images = True
        return Template._load_image(image, load_images)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'lmdeploy':
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            image = inputs.images[index]
            if self.mode == 'vllm':
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<ref>{ref}</ref>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<box>{self._get_bbox_str(bbox)}</box>']


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_vl, template_cls=QwenVLTemplate))


class QwenAudioTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        audios = inputs.audios
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _tokenize(self, context, **kwargs):
        audio_info = self.processor.process_audio(context)
        return super()._tokenize(context, audio_info=audio_info)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        text = ''.join([f'<audio>{audio}</audio>' for audio in inputs.audios])
        audio_info = self.processor.process_audio(text)
        if audio_info:
            tokenizer_kwargs = {'audio_info': audio_info}
            encoded.update(tokenizer_kwargs)
            encoded['tokenizer_kwargs'] = tokenizer_kwargs
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen_audio, template_cls=QwenAudioTemplate))


class Qwen2AudioTemplate(Template):

    def init_env_args(self) -> None:
        super().init_env_args()
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'audio'
        if not self.use_chat_template:
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']
        else:
            return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.audios:
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=self.sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=self.sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplate))


class Qwen2VLTemplate(Template):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True
    support_padding_free = True

    def init_env_args(self):
        super().init_env_args()
        self.transformers_version = version.parse(transformers.__version__)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'v3' else {}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]}, **kwargs)
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            if self.version == 'v3':
                kwargs['return_video_metadata'] = True
            video = inputs.videos[index]
            video, video_kwargs = fetch_video({'video': video}, return_video_sample_fps=True, **kwargs)
            if self.version == 'v2_5':
                inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
                tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            elif self.version == 'v3':
                if video is not None:
                    video, video_metadata = video
                    inputs.mm_processor_kwargs.setdefault('video_metadata', []).append(video_metadata)
                inputs.mm_processor_kwargs['do_sample_frames'] = False
                tokens = ['<|video_pad|>']
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|object_ref_start|>{ref}<|object_ref_end|>']

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        fps = inputs.mm_processor_kwargs['fps']
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / tmp for tmp in fps
                        ]
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def forward_context(self, model, inputs):
        if not self.padding_free or self.transformers_version >= version.parse('4.53.0.dev'):
            return super().forward_context(model, inputs)
        text_position_ids = inputs['text_position_ids']
        if self.version == 'v2':
            from transformers.models.qwen2_vl import modeling_qwen2_vl as modeling_module
        elif self.version == 'v2_5':
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as modeling_module
        elif self.version == 'omni_v2_5':
            from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni as modeling_module
        return self._patch_flash_attention_forward(modeling_module, text_position_ids)

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        return res

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
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self.model)
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        else:
            get_rope_index = base_model.model.get_rope_index
        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=attention_mask,
            **kwargs)
        return self._concat_text_position_ids(position_ids)

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            if self.transformers_version >= version.parse('4.53.0.dev') and text_position_ids.shape[0] == 1:
                # https://github.com/huggingface/transformers/pull/40194
                res.update(get_packed_seq_params(text_position_ids))
        return res


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_vl, template_cls=Qwen2VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.qvq,
        default_system=('You are a helpful and harmless assistant. You are Qwen developed by Alibaba. '
                        'Answer in the language of the question. You should think step-by-step.'),
        template_cls=Qwen2VLTemplate,
    ))


class Qwen2_5VLTemplate(Qwen2VLTemplate):
    version = 'v2_5'
    norm_bbox = 'none'


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_vl, template_cls=Qwen2_5VLTemplate))

register_template(
    QwenTemplateMeta(
        MLLMTemplateType.mimo_vl,
        template_cls=Qwen2_5VLTemplate,
        default_system='You are MiMo, an AI assistant developed by Xiaomi.'))


class Qwen3VLTemplate(Qwen2VLTemplate):
    version = 'v3'

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        do_resize=False,
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length)
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(QwenTemplateMeta(MLLMTemplateType.qwen3_vl, template_cls=Qwen3VLTemplate, default_system=None))


class Qwen2_5OmniTemplate(Qwen2_5VLTemplate):
    version = 'omni_v2_5'
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        if processor is None:
            return
        super().init_processor(processor)
        if self.version == 'omni_v2_5':
            from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
            default = Qwen2_5OmniProcessorKwargs._defaults
        elif self.version == 'omni_v3':
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
            default = Qwen3OmniMoeProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            if self.version == 'omni_v2_5':
                return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
            elif self.version == 'omni_v3':
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            if self.version == 'omni_v2_5':
                return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
            elif self.version == 'omni_v3':
                return ['<|audio_start|><|audio_pad|><|audio_end|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            inputs.videos[index] = fetch_video({'video': video}).to(torch.uint8)
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                if self.version == 'omni_v2_5':
                    return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
                elif self.version == 'omni_v3':
                    return ['<|vision_start|><|audio_start|><|video_pad|><|audio_end|><|vision_end|>']
            if self.version == 'omni_v2_5':
                return ['<|vision_bos|><|VIDEO|><|vision_eos|>']
            elif self.version == 'omni_v3':
                return ['<|vision_start|><|video_pad|><|vision_end|>']

    def _get_feat_extract_output_lengths(self, input_lengths):
        if self.version == 'omni_v2_5':
            return ((input_lengths - 1) // 2 + 1 - 2) // 2 + 1
        elif self.version == 'omni_v3':
            input_lengths_leave = input_lengths % 100
            feat_lengths = (input_lengths_leave - 1) // 2 + 1
            return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _get_new_tokens_use_audio_in_video(self, i, *, video_grid_thw, video_second_per_grid, audio_lengths,
                                           video_token_id, audio_token_id):
        merge_size = self.processor.image_processor.merge_size
        grid_thw = video_grid_thw[i]
        height = grid_thw[1] // merge_size
        width = grid_thw[2] // merge_size
        audio_token_indices = torch.arange(audio_lengths[i])
        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)

        video_token_indices = torch.broadcast_to(video_token_indices,
                                                 (video_token_indices.shape[0], height, width)).reshape(-1)
        video_token_indices = (video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
        if self.version == 'omni_v2_5':
            tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
            video_chunk_indexes = self.processor.get_chunked_index(video_token_indices, tokens_per_chunk)
            audio_chunk_indexes = self.processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

            res = []
            for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                if j < len(video_chunk_indexes):
                    video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                    res += video_token_id * video_seq_length
                if j < len(audio_chunk_indexes):
                    audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                    res += audio_token_id * audio_seq_length
            return res
        elif self.version == 'omni_v3':
            res = []
            video_data_index, audio_data_index = 0, 0
            while video_data_index < len(video_token_indices) and audio_data_index < len(audio_token_indices):
                if video_token_indices[video_data_index] <= audio_token_indices[audio_data_index]:
                    res += video_token_id
                    video_data_index += 1
                else:
                    res += audio_token_id
                    audio_data_index += 1
            if video_data_index < len(video_token_indices):
                res += video_token_id * (len(video_token_indices) - video_data_index)
            if audio_data_index < len(audio_token_indices):
                res += audio_token_id * (len(audio_token_indices) - audio_data_index)
            return res

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        processor = self.processor
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        do_resize = self.version == 'omni_v3'
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=do_resize,
            return_tensors='pt')
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        config = self.config.thinker_config
        # audio
        if self.version == 'omni_v3':
            audio_token_id = [config.audio_token_id]
        else:
            audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = self._get_feat_extract_output_lengths(audio_feature_lengths)
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        if idx_list:
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        for media_type in ['image', 'video']:
            if self.version == 'omni_v3':
                token_id = [getattr(config, f'{media_type}_token_id')]
            else:
                token = f'<|{media_type.upper()}|>'
                token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']
                    _get_new_tokens_use_audio_in_video = partial(
                        self._get_new_tokens_use_audio_in_video,
                        video_grid_thw=media_grid_thw,
                        video_second_per_grid=video_second_per_grid,
                        audio_lengths=audio_lengths,
                        video_token_id=token_id,
                        audio_token_id=audio_token_id)
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')
        feature_attention_mask = inputs.get('feature_attention_mask')

        base_model = self.get_base_model(model)
        inputs_embeds = base_model.thinker.model.embed_tokens(input_ids)
        thinker_config = model.config.thinker_config
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.thinker.visual, self.processor,
                                                   thinker_config)
        if input_features is None:
            if is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
                # Note: ZeRO-3 still results in hangs; for audio training, please use ZeRO-2.
                input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
                feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
                audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
                inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _get_position_ids(self, inputs: Dict[str, Any]):
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return self._concat_text_position_ids(position_ids)

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            if self.version == 'omni_v3':
                max_length = max(input_feature.shape[-1] for input_feature in input_features)
                for i, input_feature in enumerate(input_features):
                    mask = feature_attention_mask[i]
                    input_features[i] = F.pad(input_feature, (0, max_length - input_feature.shape[-1]))
                    feature_attention_mask[i] = F.pad(mask, (0, max_length - mask.shape[-1]))
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(QwenTemplateMeta(MLLMTemplateType.qwen2_5_omni, template_cls=Qwen2_5OmniTemplate))


class Qwen3OmniTemplate(Qwen2_5OmniTemplate, ThinkingTemplate):
    version = 'omni_v3'
    norm_bbox = 'norm1000'
    placeholder_tokens = ['<|image_pad|>', '<|audio_pad|>', '<|video_pad|>']

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_template(QwenTemplateMeta(MLLMTemplateType.qwen3_omni, template_cls=Qwen3OmniTemplate, default_system=None))


class Ovis1_6Template(Template):
    skip_prompt = False
    use_model = True

    def init_env_args(self):
        super().init_env_args()
        self.max_partition = get_env_args('max_partition', int, 9)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        idx_list = findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=self.max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        dtype = self.model.visual_tokenizer.dtype
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype)
        else:
            pixel_values = torch.zeros((1, 3, 384, 384), dtype=dtype)  # dummpy
        encoded.update({'input_ids': input_ids, 'labels': labels})
        encoded['pixel_values'] = [pixel_values]
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        padding_side = self.padding_side if self.is_training else 'left'
        if self.max_length is not None:
            model.config.multimodal_max_length = self.max_length
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')
        if labels is None:
            labels = input_ids.new_full(input_ids.shape, -100)
        _, inputs_embeds, labels, attention_mask = model.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=torch.ones_like(input_ids),  # not use, only compat
            text_labels=labels,
            pixel_values=inputs['pixel_values'],
            left_padding=padding_side == 'left')
        if inputs.get('labels') is None:
            labels = None
        return {'inputs_embeds': inputs_embeds, 'labels': labels, 'attention_mask': attention_mask}

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = self.gather_list(batch, 'pixel_values')
        res = super()._data_collator(batch, padding_to=padding_to)
        res['pixel_values'] = pixel_values
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.ovis1_6,
        prefix=['<bos>'],
        prompt=['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
        chat_sep=['<end_of_turn>\n'],
        suffix=['<end_of_turn>'],
        system_prefix=['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'],
        template_cls=Ovis1_6Template,
    ))

register_template(
    Llama3TemplateMeta(
        MLLMTemplateType.ovis1_6_llama3,
        default_system='You are a helpful and honest multimodal assistant.',
        template_cls=Ovis1_6Template,
    ))


class Ovis2Template(Ovis1_6Template):
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    NFRAMES = 12

    def init_env_args(self):
        super().init_env_args()
        self.nframes = get_env_args('nframes', int, self.NFRAMES)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image>\n']
            return [[-200], '\n']
        elif media_type == 'video':
            inputs.images = load_video_ovis2(inputs.videos[index], self.nframes)
            return [[-200] * self.nframes, '\n']


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2,
    template_cls=Ovis2Template,
))


class Ovis2_5Template(ThinkingTemplate):
    use_model = True
    skip_prompt = False
    support_padding_free = True

    def init_env_args(self) -> None:
        super().init_env_args()
        self.min_pixels = get_env_args('min_pixels', int, 448 * 448)
        self.max_pixels = get_env_args('max_pixels', int, 1344 * 1792)
        self.video_max_pixels = get_env_args('video_max_pixels', int, 896 * 896)
        self.num_frames = get_env_args('num_frames', int, 8)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image>']
            else:
                return [[-200], '\n']
        elif media_type == 'video':
            if self.mode == 'vllm':
                return ['<video>']
            else:
                inputs.images = load_video_ovis2_5(inputs.videos[index], self.num_frames)
                return [[-200], '\n']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        input_ids = encoded['input_ids']
        visual_tokenizer = self._get_model().visual_tokenizer
        idx_list = findall(input_ids, [-200])
        if inputs.videos:
            assert len(inputs.videos) == 1, 'only support single video'
            encoded['pixel_values'], encoded['grid_thws'] = visual_tokenizer.preprocess(
                video=inputs.images, min_pixels=self.min_pixels, max_pixels=self.video_max_pixels)
            num_video_tokens = encoded['grid_thws'].prod(dim=-1)
            num_video_tokens //= visual_tokenizer.vit.config.hidden_stride**2
            num_video_tokens //= visual_tokenizer.vit.config.temporal_patch_size

            def _get_new_tokens(i):
                token_len = num_video_tokens[i].item()
                return [-303] + [-300] * token_len + [-304]

            input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list, _get_new_tokens)
        elif images:
            pixel_values, grid_thws = zip(
                *(visual_tokenizer.preprocess(image=image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                  for image in images))
            encoded['pixel_values'] = torch.cat(pixel_values, dim=0)
            encoded['grid_thws'] = torch.cat(grid_thws, dim=0)

            num_image_atoms = encoded['grid_thws'].prod(dim=-1)
            num_image_atoms //= visual_tokenizer.vit.config.hidden_stride**2
            num_image_atoms //= visual_tokenizer.vit.config.temporal_patch_size

            def _get_new_tokens(i):
                token_len = num_image_atoms[i].item()
                return [-301] + [-300] * token_len + [-302]

            input_ids, encoded['labels'], encoded['loss_scale'] = self._extend_tokens(
                input_ids, encoded['labels'], encoded['loss_scale'], idx_list, _get_new_tokens)

        encoded['input_ids'] = input_ids
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values', None)
        grid_thws = inputs.get('grid_thws')
        INDICATOR_IDS = [-301, -302, -303, -304]
        VISUAL_ATOM_ID = -300
        placeholder_token_mask = torch.lt(input_ids, 0)
        inputs_embeds = model.get_wte()(torch.masked_fill(input_ids, placeholder_token_mask, 0))

        if pixel_values is not None or is_deepspeed_enabled():
            visual_indicator_embeds = model.vte(model.indicator_token_indices).to(
                dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            for i, indicator_id in enumerate(INDICATOR_IDS):
                inputs_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
        if pixel_values is not None:
            visual_tokens = model.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = model.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
        elif is_deepspeed_enabled():
            pixel_values, grid_thws = model.visual_tokenizer.preprocess(
                Image.new('RGB', (32, 32), (0, 0, 0)), min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            pixel_values = pixel_values.to(device=inputs_embeds.device)
            grid_thws = grid_thws.to(device=inputs_embeds.device)
            visual_tokens = model.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = model.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = inputs_embeds + visual_embeds.mean() * 0.

        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        grid_thws = self.concat_tensor(batch, 'grid_thws', 0)
        if grid_thws is not None:
            res['grid_thws'] = grid_thws
        return res


register_template(QwenTemplateMeta(
    MLLMTemplateType.ovis2_5,
    template_cls=Ovis2_5Template,
    default_system=None,
))


@dataclass
class MarcoO1TemplateMeta(QwenTemplateMeta):
    default_system: Optional[str] = """
你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.
        \n## 重要！！！！！
当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。
<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。
        """


register_template(MarcoO1TemplateMeta(LLMTemplateType.marco_o1))
