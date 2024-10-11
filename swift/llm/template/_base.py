# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
import torch
import torch.nn as nn
from modelscope import get_logger
from PIL import Image
from transformers import PreTrainedTokenizerBase

from ..infer import InferRequest
from ..utils import Messages, decode_base64
from .agent import get_tools_prompt, loss_scale_map, split_str_parts_by
from .utils import Context, ContextType, Prompt, StopWords, fetch_one
from .vision_utils import load_batch, load_image, rescale_image, normalize_bbox

logger = get_logger()


@dataclass
class TemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    system: Optional[str] = None  # If it is None, set it to template.default_system.

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.image_idx = 0
        self.audio_idx = 0
        self.video_idx = 0
        self.object_idx = 0
        self.box_idx = 0

    def copy(self):
        return TemplateInputs(deepcopy(self.messages), self.system, self.images.copy(),
                              self.audios.copy(), self.videos.copy(), self.objects.copy())

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)


    @staticmethod
    def from_infer_request(request: InferRequest, *, tools_prompt: str = 'react_en') -> 'TemplateInputs':
        request = request.copy()
        messages = request.messages
        assert len(messages) >= 1

        tools = request.tools
        if messages[0]['role'] == 'system':
            message = messages.pop(0)
            system = message['content']
        else:
            system = None

        if tools is not None:
            assert system is None
            if isinstance(tools, str):
                tools = json.loads(tools)
            system = get_tools_prompt(tools, tools_prompt)

        media_kwargs = InferRequest.remove_messages_media(request.messages)
        inputs = TemplateInputs(messages, system, **media_kwargs, objects=request.objects)
        return inputs


class Template:
    """A template class for all supported models.

    Args:
        prefix: Prefix before the first round
        prompt: Template format for each round
        chat_sep: Separator symbol between rounds. If set to None, it is a single-round template and
            does not support multi-round conversations.
        suffix: Ending for the last round, used to stop generation.
        default_system: Default system
        system_prefix: If it includes a prefix for the system, used in cases where the prefix parameter
            cannot accommodate a template with the system.
        tool_prompt: The prompt when the role of messages is set to 'tool'.

        stop_words: A list of stop words, where each stop word can consist of multiple tokens.
        placeholder_tokens: A list of placeholder tokens, where each placeholder token can only be a single token.
        auto_add_bos: By default, the bos_token is not added. The auto_add_bos option will determine
            whether to add it based on `tokenizer.encode('')`.
        tools_prompt: The type of tools_prompt added in the system.

    Examples:
        chatml (with bos):
            prefix: <s>
            prompt: <|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n
            chat_sep: <|im_end|>\n
            suffix: <|im_end|>
            system_prefix: <s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n

        <s><|im_start|>system  # prefix or system_prefix
        {{SYSTEM}}<|im_end|>
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # chat_sep
        <|im_start|>user  # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>  # suffix
    """

    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
    special_keys = ['images', 'videos', 'audios', 'objects']
    grounding_type = 'norm_1000'
    image_placeholder = ['<image>']
    load_medias = True

    loss_scale = 'default'
    output_prompt_answer = False  # for encoder-decoder & kto
    padding_side: Literal['left', 'right'] = 'right'  # The padding_side when the training batch_size >= 2.

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 system_prefix: Optional[Prompt] = None,
                 tool_prompt: Optional[Prompt] = None,
                 *,
                 stop_words: Optional[StopWords] = None,
                 placeholder_tokens: Union[int, str, None] = None,
                 auto_add_bos: bool = False,
                 tools_prompt: str = 'react_en',
                 skip_input_len: bool = True) -> None:
        # check
        if default_system is None:
            default_system = ''
        for x in [prefix, prompt, chat_sep, suffix, system_prefix]:
            assert x is None or isinstance(x, list)

        if self._has_system(prefix):
            assert system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            system_prefix = prefix
            prefix = self._replace_system(prefix)
        self.is_post_system = self._has_system(prompt)  # mistral_nemo
        if self.is_post_system:
            self.prompt = [context for context in prompt if '{{SYSTEM}}' not in context]
            self.system_prompt = prompt
        else:
            self.prompt = prompt
        if system_prefix is None and not self.is_post_system:
            self.support_system = False
            assert not default_system, 'The template does not support `system`.'
        else:
            self.support_system = True
        self.support_multi_round = chat_sep is not None

        self.prefix = prefix
        self.system_prefix = system_prefix
        self.chat_sep = chat_sep
        self.suffix = suffix
        self.default_system = default_system
        self.tool_prompt = tool_prompt if tool_prompt is not None else prompt  # default as user

        self.stop_words = stop_words
        self.placeholder_tokens = placeholder_tokens
        self.auto_add_bos = auto_add_bos
        self.tools_prompt = tools_prompt
        self.skip_input_len = skip_input_len
        self._is_init = False

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        """Replace system with the """
        return [p.replace('{{SYSTEM}}', '') for p in prefix]

    @staticmethod
    def _has_system(prefix_or_prompt: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix_or_prompt])

    @staticmethod
    def _token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Turn `eos_token_id` to token id

        e.g. [['eos_token_id']] -> [[2]]
        """
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                v = [getattr(tokenizer, sub_v) if isinstance(sub_v, str) else sub_v for sub_v in v]
            res_value.append(v)
        return res_value

    def _check_system(self, system: str) -> str:
        if system is None:
            system = ''
        if system:
            assert self.support_system, (f'The template does not support `system`, template_type: {self.template_type}')
        return system

    def _init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       *,
                       truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
                       loss_scale: Optional[str] = None,
                       max_image_size: int = -1) -> None:
        """
        default_system: Override the default_system in the template.
        max_length: Max length of the sequence
        truncation_strategy: The truncation strategy
        loss_scale: The loss scale function to use
        max_image_size: Rescale image to reduce memory usage, default `-1` means no limitation.
            e.g. 512 * 512 (H*W)
        """
        assert self._is_init is False, 'The template has been initialized.'
        self._is_init = True

        # if default_system is None. not change self.default_system
        if default_system is not None:
            self.default_system = self._check_system(default_system)

        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self._token_attr_to_id(tokenizer, value)
            setattr(self, key, value)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        if loss_scale:
            self.loss_scale = loss_scale
        self.max_image_size = max_image_size
        self.is_multimodal = getattr(tokenizer, 'is_multimodal', None)
        self.task: Literal['train', 'infer_pt', 'infer_vllm', 'infer_lmdeploy'] = 'infer_pt'


    def _preprocess_inputs(self, inputs: TemplateInputs, *, max_image_size: int = -1,) -> None:
        system = inputs.system
        if system is None:
            system = self.default_system
        inputs.system = self._check_system(system)

        self._get_std_messages(inputs.messages)
        n_round = len(inputs.messages) // 2
        if n_round > 1 and not self.support_multi_round:
            raise ValueError(f'The template does not support multi-round chat, template_type: {self.template_type}')

        images = inputs.images
        if images and self.load_medias:
            images = load_batch(images, load_image)
            if max_image_size != -1:
                assert self.grounding_type != 'real', 'not support'  # TODO:check
                images = [rescale_image(img, max_image_size) for img in images]
            inputs.images = images
        if inputs.objects:
            self._preprocess_objects(inputs, inputs.objects)

        if inputs.is_multimodal:
            self._add_default_tags(inputs)

    def encode(
        self,
        inputs: Union[InferRequest, TemplateInputs],
    ) -> Dict[str, Any]:
        """The entrance method of Template!

        Returns:
            return {'input_ids': List[int], 'labels': Optional[List[int]], ...}
        """
        # Template needs to be initialized
        if not self._is_init:
            raise ValueError(
                'Template is not initialized, please use the `get_template` function to obtain the template.')

        if isinstance(inputs, InferRequest):
            inputs = TemplateInputs.from_infer_request(inputs, tools_prompt=self.tools_prompt)
        else:
            inputs = inputs.copy()
        assert isinstance(inputs, TemplateInputs)
        self._preprocess_inputs(inputs)
        if self.task in {'train', 'infer_pt'}:
            self._check_inputs(inputs)
            res = self._encode(inputs)
        else:
            res = Template._encode(self, inputs)
            if inputs.images:
                res['images'] = inputs.images
        if self.task != 'train':
            res.pop('loss_scale', None)
        return res

    def post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def _preprocess_objects(self, inputs: TemplateInputs, objects: List[Dict[str, Any]]):
        # Load image into PIL format
        images = inputs.images
        images = load_batch(images, load_image)  # base64/local_path -> PIL.Image
        # Normalize grounding bboxes
        normalize_bbox(objects, images, to_type=self.grounding_type)
        if not self.load_medias:  # fix pt & qwen-vl
            images = decode_base64(images=images)['images']  # PIL.Image/base64 -> local_path
        inputs.images = images
        inputs.objects = objects

    @staticmethod
    def _concat_context_list(
            context_list: List[Context],
            res_context_list: List[Context],  # inplace
            res_context_type: List[ContextType],  # inplace
            system: Optional[str] = None,
            query: Optional[str] = None,
            response: Optional[str] = None,
            round0: Optional[int] = None) -> None:
        """Concat context list and replace placeholder"""
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    context = response
                    res_context_list.append(response)
                    res_context_type.append(ContextType.RESPONSE)
                    continue
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        assert isinstance(new_str, str), f'new_str: {new_str}'
                        context = context.replace(old_str, new_str)
            if len(context) == 0:
                continue
            res_context_list.append(context)
            res_context_type.append(ContextType.OTHER)

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               inputs: TemplateInputs) -> Tuple[List[Context], List[float]]:
        """Merge anything in the context to simplify the inputs"""
        if inputs.is_multimodal:
            context_list, loss_scale_list = self._split_special_tokens(context_list, loss_scale_list)
        context_list, loss_scale_list = self._pre_tokenize(context_list, loss_scale_list, inputs)

        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list
        temp: List[str] = []
        temp_loss_scale = 0.
        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str) and (loss_scale == temp_loss_scale):
                temp.append(context)
            else:
                if len(temp) > 0:
                    res.append(''.join(temp))
                    res_loss_scale.append(temp_loss_scale)
                    temp.clear()
                if isinstance(context, str):  # loss_scale diff
                    temp.append(context)
                else:
                    res.append(context)
                    res_loss_scale.append(loss_scale)
                temp_loss_scale = loss_scale
        if len(temp) > 0:
            res.append(''.join(temp))
            res_loss_scale.append(temp_loss_scale)

        return res, res_loss_scale

    @staticmethod
    def _split_special_tokens(context_list: List[Context],
                              loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        """Split special tokens, for example `<image>`, `<video>`, this will help the replace_tag operation"""
        res: List[Context] = []
        loss_scale_res: List[float] = []
        for context, loss_scale in zip(context_list, loss_scale_list):
            contexts = []
            if isinstance(fetch_one(context), str):
                for d in split_str_parts_by(context, Template.special_tokens):
                    contexts.extend([d['key'], d['content']])
                contexts = [c for c in contexts if c]
                res.extend(contexts)
                loss_scale_res.extend([loss_scale] * len(contexts))
            else:
                res.append(context)
                loss_scale_res.append(loss_scale)
        return res, loss_scale_res

    def _tokenize(self, context, **tokenizer_kwargs):
        return self.tokenizer(
            context, return_attention_mask=False, add_special_tokens=False, **tokenizer_kwargs)['input_ids']

    def _check_inputs(self, inputs: TemplateInputs) -> None:
        """Check inputs valid"""
        pass

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: TemplateInputs) -> List[Context]:
        """Override this function to do your own replace operation.

        This method is used to replace standard tags like `<image>` to some tokens that the model needs.

        Args:
            media_type: The modal.
            index: The index of the medias, for index 0 represents the first elements in `images`
            inputs: The inputs

        Returns:
            The content or input_ids after replacement.
        """
        if media_type == 'image':
            return self.image_placeholder
        elif media_type == 'video':
            return ['<video>']
        elif media_type == 'audio':
            return ['<audio>']

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: TemplateInputs) -> List[Context]:
        """Replace objects referenced by the bbox to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            object_: inputs.objects[inputs.object_idx]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [object_['caption']]

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: TemplateInputs) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            object_: inputs.objects[inputs.object_idx]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                all_objects += f'[({sub_object[0]},{sub_object[1]}),' f'({sub_object[2]},{sub_object[3]})],'
            all_objects = all_objects[:-1]
            return [all_objects]
        else:
            return [f'[({object_["bbox"][0]},{object_["bbox"][1]}),({object_["bbox"][2]},{object_["bbox"][3]})]']


    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: TemplateInputs) -> Tuple[List[Context], List[float]]:
        """This method happens before tokenization, replace standard tags to the contents or input_ids needed by
        the model.

        Args:
            context_list: The content list
            loss_scale_list: The loss scale list
        Returns:
            The context_list and loss_scale_list after replacement.
        """
        # replace tag/object/box
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        # reset
        for k in ['image', 'video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['image', 'video', 'audio']:
                if context == f'<{k}>':
                    idx = getattr(inputs, f'{k}_idx')
                    c_list = self.replace_tag(k, idx, inputs)
                    setattr(inputs, f'{k}_idx', idx + 1)
                    break
            else:
                if context == '<ref-object>':
                    idx = inputs.object_idx
                    c_list = self.replace_object(inputs.objects[idx], idx, inputs)
                    inputs.object_idx = idx + 1
                elif context == '<bbox>':
                    idx = inputs.object_idx
                    c_list = self.replace_box(inputs.objects[idx], idx, inputs)
                    inputs.box_idx = idx + 1
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    @staticmethod
    def _add_default_tags(inputs: TemplateInputs):
        total_content = '\n'.join([message['content'] for message in inputs.messages])
        for media_type in ['image', 'audio', 'video']:
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            medias = getattr(inputs, media_key)
            if medias:
                num_media_tags = len(re.findall(media_tag, total_content))
                num_media = len(medias)
                num_new_tags = num_media - num_media_tags
                assert num_new_tags >= 0, f'Number of media: {num_media}, number of media_tags: {num_media_tags}'
                inputs.messages[0]['content'] = media_tag * num_new_tags + inputs.messages[0]['content']

    def _encode_context_list(
            self,
            context_list: List[Context],
            loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """return: input_ids, labels, tokenizer_kwargs"""
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        tokenizer_kwargs = {}
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                # tokenizer_kwargs is the returned tokenizer_kwargs,
                # while curr_tokenizer_kwargs is the tokenizer_kwargs for the current context.
                curr_tokenizer_kwargs = self._get_tokenizer_kwargs(context)
                self._concat_tokenizer_kwargs(tokenizer_kwargs, curr_tokenizer_kwargs)
                token_list = self._tokenize(context, **curr_tokenizer_kwargs)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            loss_scale.extend([loss_weight] * len(token_list))
        return input_ids, labels, loss_scale, tokenizer_kwargs

    @staticmethod
    def _use_dynamic_eos(labels: List[int], suffix_tokens_id: List[int]) -> None:
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels)):
            if labels[i - 1] >= 0 and labels[i] == -100:
                start = i
            if start > 0 and labels[i - 1] == -100 and labels[i] >= 0:
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len:
                    labels[start:start + suffix_len] = suffix_tokens_id

    @staticmethod
    def _get_std_messages(messages):
        if messages[0]['role'] == 'response':
            messages.insert(0, {'role': 'query', 'content': ''})  # pretrain
        if len(messages) % 2 == 1:
            messages.append({'role': 'assistant', 'content': None})  # inference

    def _encode(self, inputs: TemplateInputs) -> Dict[str, Any]:

        res_context_list: List[Context] = []
        res_context_types: List[ContextType] = []
        if self.auto_add_bos:
            bos_token_id = self.tokenizer.bos_token_id
            if isinstance(bos_token_id, int) and bos_token_id in self.tokenizer.encode(''):
                res_context_list.append([bos_token_id])
                res_context_types.append(ContextType.OTHER)

        prefix = self.system_prefix if inputs.system else self.prefix
        self._concat_context_list(prefix, res_context_list, res_context_types, system=inputs.system)

        n_round = len(inputs.messages) // 2
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']
            response_role, response = response_message['role'], response_message['content']
            # TODO: Optimize the Template mechanism.
            assert query_role in {'user', 'tool'}
            assert response_role in {'assistant'}
            if query_role == 'tool':
                prompt = self.tool_prompt
            elif self.is_post_system and i == n_round - 1:
                prompt = self.system_prompt
            else:
                prompt = self.prompt

            context_list = prompt.copy()
            extra_context_list = []
            extra_context_type = None
            if i < n_round - 1:
                context_list.append('{{RESPONSE}}')
                extra_context_list = self.chat_sep  # TODO:agent check
                extra_context_type = ContextType.OTHER
            elif response is not None:
                # It is the final round, and the response exists (during training).
                context_list.append('{{RESPONSE}}')
                extra_context_list = self.suffix
                extra_context_type = ContextType.SUFFIX
            assert query or response  # TODO:check
            self._concat_context_list(
                context_list,
                res_context_list,
                res_context_types,
                query=query,
                response=response,
                system=inputs.system,
                round0=i)
            res_context_list += extra_context_list
            res_context_types += [extra_context_type] * len(extra_context_list)
        loss_scale_list = loss_scale_map[self.loss_scale](res_context_types, inputs.messages)

        res = {}
        if self.output_prompt_answer:
            # tokenizer_kwargs: use prompt (qwen-audio)
            answer_len = len(extra_context_list) + bool(response is not None)
            total_len = len(res_context_list)
            for key, _slice in zip(['answer', 'prompt'],
                                   [slice(total_len - answer_len, total_len),
                                    slice(0, total_len - answer_len)]):
                _res_context_list, _loss_scale_list = self._simplify_context_list(res_context_list[_slice],
                                                                                  loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                    _res_context_list, _loss_scale_list)
                res[f'{key}_input_ids'], res[f'{key}_labels'] = input_ids, labels
                if self.loss_scale != 'default':
                    res[f'{key}_loss_scale'] = loss_scale
            input_ids = res['prompt_input_ids'] + res['answer_input_ids']
            labels = res['prompt_labels'] + res['answer_labels']
            if response is None:
                assert len(res['answer_labels']) == 0
                res['answer_labels'] = None
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)
            self._use_dynamic_eos(labels, self._encode_context_list(self.suffix)[0])

        if tokenizer_kwargs:
            res['tokenizer_kwargs'] = tokenizer_kwargs

        if response is None:
            labels = None

        if self.max_length is not None:
            if self.truncation_strategy == 'delete' and len(input_ids) > self.max_length:
                logger.warn(f'Current length of row({len(input_ids)}) is larger'
                            f' than the max_length({self.max_length}), deleted.')
                return {}
            input_ids = input_ids[-self.max_length:]
            if labels is not None:
                labels = labels[-self.max_length:]
            if loss_scale is not None:
                loss_scale = loss_scale[-self.max_length:]
        res['input_ids'] = input_ids
        res['labels'] = labels
        res['loss_scale'] = loss_scale
        return res

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        assert len(tokenizer_kwargs) == 0

    def get_generate_ids(self, generate_ids: torch.Tensor, input_token_len: int) -> List[int]:
        if isinstance(generate_ids, torch.Tensor):
            generate_ids = generate_ids.tolist()
        if len(generate_ids) > 0 and not isinstance(generate_ids[0], (list, tuple)):
            generate_ids = generate_ids[0]  # to 1d list
        if self.skip_input_len:
            return generate_ids[input_token_len:]
        else:
            return generate_ids

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # copy from transformers.generation.streamers.TextStreamer
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    @staticmethod
    def _get_safe_print_idx(response: str, print_idx: int, is_finished: bool = False) -> int:
        if is_finished:
            return len(response)
        if response.endswith('\n') or len(response) > 0 and Template._is_chinese_char(ord(response[-1])):
            print_idx = len(response)
        else:
            print_idx = max(response.rfind(' ') + 1, print_idx)
        return print_idx

    def generate_ids_to_response(
        self,
        generate_ids: List[int],
        is_finished: bool = True,
        *,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        # only stream=True
        return_delta: bool = False,
        print_idx: Optional[List[int]] = None,
        first_num_space: Optional[List[int]] = None,
    ):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer = self.tokenizer
        if hasattr(generate_ids, 'tolist'):
            generate_ids = generate_ids.tolist()
        # avoid printing template.suffix[-1])
        if isinstance(self.suffix[-1], list) and (not is_finished or is_finished
                                                  and generate_ids[-len(self.suffix[-1]):] == self.suffix[-1]):
            generate_ids = generate_ids[:-len(self.suffix[-1])]
        if not is_finished or is_finished and generate_ids[-1:] == [self.tokenizer.eos_token_id]:
            generate_ids = generate_ids[:-1]
        response = tokenizer.decode(generate_ids, **tokenizer_kwargs)
        if first_num_space is not None:
            # Avoid the occurrence of repeated words in sentence.
            res_fns = first_num_space  # res_first_num_space
            first_num_space = first_num_space[0]
            cur_num_space = len(response) - len(response.lstrip(' '))
            if not is_finished and first_num_space == -1:
                first_num_space = cur_num_space
                res_fns[0] = first_num_space
            if cur_num_space < first_num_space:
                response = ' ' * (first_num_space - cur_num_space) + response
            elif cur_num_space > first_num_space:
                response = response[cur_num_space - first_num_space:]
        if isinstance(self.suffix[-1],
                      str) and (not is_finished or is_finished and response[-len(self.suffix[-1]):] == self.suffix[-1]):
            idx = max(len(response) - len(self.suffix[-1]), 0)
            # To avoid response length being shorter than previous response length during streaming.
            if print_idx is not None:
                idx = max(idx, print_idx[0])
            response = response[:idx]

        if print_idx is not None:
            old_print_idx = print_idx[0]
            if not is_finished:
                # avoid printing incomplete words
                print_idx[0] = self._get_safe_print_idx(response, print_idx[0])
                response = response[:print_idx[0]]
            if return_delta:
                response = response[old_print_idx:]
        else:
            assert is_finished and not return_delta
        return response

    def post_process_generate_response(self, response: str, inputs: TemplateInputs) -> str:
        return response
