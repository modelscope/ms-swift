# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from modelscope import get_logger
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, StoppingCriteria
from .loss_scale import loss_scale_map
from .tools_prompt import get_tools_prompt
from .utils import load_batch, load_image, rescale_image, fetch_one, to_device, decode_base64
from .utils import History, Prompt, StopWords, Context, Messages

logger = get_logger()

DEFAULT_SYSTEM = 'You are a helpful assistant.'

TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
    **kwargs,
) -> 'Template':
    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template.init_template(tokenizer, default_system, max_length, truncation_strategy, **kwargs)
    return template


def _findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
    """Find the index of a token in the token_list."""
    if isinstance(sub_token_list, int):
        sub_token_list = [sub_token_list]
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(sub_token_list[0], idx + 1)
            if len(sub_token_list) == 1 or sub_token_list == token_list[idx:idx + len(sub_token_list)]:
                res.append(idx)
    except ValueError:
        pass
    return res


def replace_img_tag(messages: Messages,
                    replace_token: str,
                    pattern=r'<img>(.+?)</img>') -> Tuple[str, History, List[str]]:
    images_path = []
    new_messages = []
    for i, m in enumerate(messages):
        m = m.copy()
        if m['content'] is None or m['role'] in ('tool', 'system', 'assistant'):
            new_messages.append(m)
        else:
            images_path += re.findall(pattern, m['content'])
            m['content'] = re.sub(pattern, replace_token, m['content'])
            new_messages.append(m)
    return messages, images_path


class StopWordsCriteria(StoppingCriteria):
    """Adding extra stop words in template to prevent unstoppable generation
        Like suffixes and chat seps in the template.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: StopWords, **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.tokenizer_kwargs = tokenizer_kwargs
        self.start_idx = -1

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if self.start_idx == -1:
            self.start_idx = len(input_ids[0]) - 1
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        # [-20:]: Assuming the end tokens do not exceed 20 tokens,
        #   to avoid input_ids being too long and affecting efficiency.
        text = tokenizer.decode(input_ids[0, self.start_idx:][-20:], **self.tokenizer_kwargs)
        for stop_word in stop_words:
            if isinstance(stop_word, str):
                if stop_word in text:
                    return True
            else:  # list
                if len(stop_word) > 0 and input_ids[0].tolist()[-len(stop_word):] == stop_word:
                    return True
        return False


class Template:
    """A template class for all supported models.

    Args:
        prefix: Prefix tokens before the first turn's prompt
        prompt: A list of elements whose types are str and list of integers. The input query part of every turn.
        chat_sep: The chat separators between every turn.
        suffix: The end tokens after the chat finished.
        default_system: A default system instruction.
        system_prefix: The prefix if the `system` is not empty.
        auto_add_bos: By default, the bos_token is not added. The auto_add_bos option will determine
            whether to add it based on `tokenizer.encode('')`.
        tools_prompt: The tools prompt name
        tool_prompt: The tool prompt, usually useful when there is a tool role
        padding_side: The padding side
        infer_media_type: The media type supported by the multi-modals
        Examples:
            <start_of_output>system\nYou are a helpful assistant!<end_of_output>\n<bos><start_of_output>Who are you?<end_of_output>\n<start_of_output>assistant:I am a robot<end_of_output>\n<start_of_output>Who are you?<end_of_output>\n<start_of_output>assistant:I am a robot<end_of_output> # noqa
                                     ----------system------------                                       ---query----                                            --response- -----chatsep-----                 ---query---                                             --response- ----suffix-----
            ----------------------------system_prefix---------------------------- ---------------------------- prompt -------------------------------------                                  ---------------------------- prompt -------------------------------------

    """

    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
    special_keys = ['images', 'videos', 'audios', 'objects']
    grounding_type = 'norm_1000'
    image_placeholder = ['<image>']
    load_medias = True
    compute_per_round_loss = True  # for rlhf
    output_prompt_answer = False  # for encoder-decoder & kto

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 system_prefix: Optional[Prompt] = None,
                 auto_add_bos: bool = False,
                 tools_prompt: str = 'react_en',
                 tool_prompt: Optional[Prompt] = None,
                 padding_side: Literal['left', 'right'] = 'right',
                 infer_media_type: Literal['interleave', 'dialogue', 'round'] = 'interleave') -> None:
        # check
        for x in [prefix, prompt, chat_sep, suffix, system_prefix]:
            assert x is None or isinstance(x, list)

        if default_system == '':
            default_system = None
        if self._has_system(prefix):
            assert system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            system_prefix = prefix
            prefix = self._replace_system(prefix)
        self.prefix = prefix
        self.system_prefix = system_prefix
        if self.system_prefix is None and not any(['{{SYSTEM}}' in context for context in prompt]):
            assert default_system is None, 'The template does not support `system`.'
        self.prompt = prompt
        self.chat_sep = chat_sep
        self.support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self.default_system = default_system
        self.use_default_system = True
        self.auto_add_bos = auto_add_bos
        self._is_init = False
        self.tools_prompt = tools_prompt
        self.tool_prompt = tool_prompt if tool_prompt is not None else self.prompt  # default as user
        self.padding_side = padding_side
        self.infer_media_type = infer_media_type

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        """Replace system with the """
        return [p.replace('{{SYSTEM}}', '') for p in prefix if '{{SYSTEM}}' in p]

    @staticmethod
    def _has_system(prefix: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix])

    @staticmethod
    def token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Turn `eos_token_id` to token id

        e.g. [['eos_token_id']] -> [[2]]
        """
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                res_v = []
                for sub_v in v:
                    if isinstance(sub_v, str):
                        sub_v = getattr(tokenizer, sub_v)
                    res_v.append(sub_v)
                v = res_v
            res_value.append(v)
        return res_value

    def init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
                       loss_scale: str = 'default',
                       rescale_image: int = -1,
                       **kwargs) -> None:
        """Init template by a tokenizer
        Args:
            tokenizer: The tokenizer to tokenize the sentence
            default_system: The default system to use if the dataset does not provide one
            max_length: Max length of the sequence
            truncation_strategy: The truncation strategy
            loss_scale: The loss scale function to use
            rescale_image: Rescale image to reduce memory usage, default `-1` means no limitation
        """
        assert self._is_init is False, 'The template has been initialized.'
        self._is_init = True
        self.tokenizer = tokenizer
        self.is_multimodal = getattr(tokenizer, 'is_multimodal', None)
        # if default_system is None. not change self.default_system
        if default_system == '':
            self.default_system = None
        elif default_system is not None:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {getattr(self, "template_type", None)}')
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        if isinstance(loss_scale, str):
            self.loss_scale = loss_scale_map.get(loss_scale, None)
        else:
            self.loss_scale = loss_scale
        self.rescale_image = rescale_image

        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self.token_attr_to_id(tokenizer, value)
            setattr(self, key, value)

    def post_encode(self, model: Module, data: Any) -> Dict[str, Any]:
        """This method will be called after data_collator and before the forward
        Args:
            data: The `_data` field from the example batch, this field should be packed manually
        Returns:
            Any extra fields need to be passed into the model.forward
        """
        return {}

    def check_example(self, example: Dict[str, Any]) -> None:
        """Check example valid"""
        pass

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        """Add default tags to example, this is for the multi-modal datasets
            1. For the round infer_media_type, this method will check the tag equals with the chat round
            2. Else, this method will try to add tags to the head of the messages
        Args:
            example: The input example
        """
        messages = example['messages']
        for media_key, media_tag in [('videos', '<video>'), ('images', '<image>'), ('audios', '<audio>')]:
            if example.get(media_key):
                _messages = [message for message in messages if message['role']!='system']
                n_round = len(_messages)
                assert n_round % 2 == 0
                history = [_messages[i:i+2] for i in range(n_round // 2)]
                if self.infer_media_type == 'round':
                    for i, h, m in zip(range(n_round // 2), history, example[media_key]):
                        num_media_tags = len(re.findall(media_tag, h[0]['content']))
                        if m:
                            assert num_media_tags <= 1, (
                                'The model includes at most one media per round. However, '
                                f'this round contains {num_media_tags} media_tags. query: {h[0]}')
                            if num_media_tags == 0:
                                h[0]['content'] = media_tag + h[0]['content']
                        else:
                            assert num_media_tags == 0, f'Missing media. query: {h[0]}'
                    example[media_key] = [m for m in example[media_key] if m]
                else:
                    num_media_tags = len(re.findall(media_tag, '\n'.join([h[0]['content'] for h in history])))
                    example[media_key] = [m for m in example[media_key] if m]
                    num_media = len(example[media_key])
                    num_new_tags = num_media - num_media_tags
                    assert num_new_tags >= 0, f'Number of media: {num_media}, number of media_tags: {num_media_tags}'
                    history[0][0]['content'] = media_tag * num_new_tags + history[0][0]['content']

    def replace_media_tags(self, example) -> None:
        """Replace the <img></img> with the images key and <image> tag

        Args:
            example: The input example
        """
        # Parse <img></img> format images and merged into images key
        if self.is_multimodal in {True, None}:  # If False, do not perform replace_img_tag
            example['messages'], images_path = replace_img_tag(
                example.get('messages'), '<image>')

            if example.get('images') and images_path:
                raise ValueError('Do not mix use the <img></img> tag and <image> tag.')
            example['images'] = example.get('images') or [] + images_path

        # audio, video
        if self.is_multimodal in {True, None}:
            for k, tag, pattern in zip(['audios', 'videos'], ['<audio>', '<video>'],
                                       [r'<audio>(.+?)</audio>', r'<video>(.+?)</video>']):
                example['messages'], medias_path = replace_img_tag(
                    example.get('messages'), tag, pattern)

                example[k] = example.get(k) or [] + medias_path

    def _preprocess_media(self, example):
        """Preprocess multi-modal media resources in one example
            1. Wrap all values in media keys to list
            2. Replace <img></img> tags
            3. Add or check missing tags to examples
            4. Parse the string field in the `objects` field to jsons
            5. Load images if needed
        Args:
            example: The input example
        """
        multimodal_keys = {
            'audio': 'audios',
            'image': 'images',
            'video': 'videos',
        }
        # Format media_keys to list
        for media_key in multimodal_keys.values():
            if example.get(media_key) and not isinstance(example[media_key], (tuple, list)):
                # change images field to list
                example[media_key] = [example[media_key]]

        self.replace_media_tags(example)
        # Add default tags to examples to note where to put the medias into the sequence
        self.add_default_tags(example)

        # Format objects(groundings/refs) to json
        if example.get('objects') and isinstance(example['objects'], str):
            # reload grounding from str
            example['objects'] = json.loads(example['objects'])
            objects = []
            for object in example['objects']:
                # Compatible with list format
                if isinstance(object, list):
                    object = {
                        'caption': object[0],
                        'bbox': object[1],
                        'bbox_type': None,
                        'image': 0,
                    }
                objects.append(object)
            example['objects'] = objects

        # Load image into PIL format
        images = example.get('images') or []
        if images:
            if example.get('objects') or self.load_medias:
                images = load_batch(images, load_image)  # base64/local_path -> PIL.Image
            if example.get('objects'):
                # Normalize grounding bboxes
                self.normalize_bbox(example['objects'], images, to_type=self.grounding_type)
            if self.load_medias and self.grounding_type != 'real':
                images = [rescale_image(img, self.rescale_image) for img in images]
            if not self.load_medias:  # fix pt & qwen-vl
                images = decode_base64(images=images)['images']  # PIL.Image/base64 -> local_path
            example['images'] = images

    def preprocess(self, example):
        # Duplicate example and create a new one to prepare in-place changes
        example = example.copy()
        template_type: Optional[str] = getattr(self, 'template_type', None)
        tools: Union[List[Any], str] = example.get('tools') or []

        # Template needs to be initialized
        if not self._is_init:
            raise ValueError(
                'Template is not initialized, please use the `get_template` function to obtain the template.')

        messages = example['messages']
        system_round = [message for message in messages if message['role'] == 'system']
        messages = [message for message in messages if message['role'] != 'system']
        # Reset system (by default value and agent tools)
        system: Optional[str] = system_round[0]['content'] if system_round else ''
        if not system:
            if self.use_default_system:
                system = self.default_system
        else:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {template_type}')
        if tools:
            if isinstance(tools, str):
                tools = json.loads(tools)
            if system is None:
                system = ''
            system += get_tools_prompt(tools, self.tools_prompt)

        if system:
            if not system_round:
                system_round = [{'role': 'system', 'content': None}]
            system_round[0]['content'] = system

        if len(messages) > 1:
            assert self.support_multi_round, (
                f'The template does not support multi-round chat, template_type: {template_type}')
        example['messages'] = system_round + messages
        self._preprocess_media(example)
        # Check the example that whether matching the very template's rules
        self.check_example(example)
        return example

    def encode(self, example: Dict[str, Any], streaming: bool = False, is_training: bool = False, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """The entrance method of Template!

        Args:
            example: The input example
            streaming: If is streaming mode
            is_training: Use template in training
            **kwargs:
                model: The model instance, use only in `is_training=False`
        Returns:
            if not streaming mode, returns tuple of (example, tokenizer_kwargs), else return example only
        """
        example = self.preprocess(example)
        res = self._encode(example, **kwargs)
        inputs = res[0]
        if not is_training and '_data' in inputs:
            model = kwargs.get('model')
            assert model is not None
            data = inputs.pop('_data')
            data = to_device(data, model.device)
            inputs.update(self.post_encode(model, data))
        return res if not streaming else inputs

    def _encode(self, example: Dict[str, Any], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """return: inputs, tokenizer_kwargs"""
        messages = example['messages']
        is_multi_modal: bool = any([example.get(key) for key in Template.special_keys])

        inputs, tokenizer_kwargs = self._concat_and_tokenize(
            messages,
            self.truncation_strategy,
            auto_add_bos=self.auto_add_bos,
            is_multi_modal=is_multi_modal,
            example=example)
        if inputs.get('labels') is None:
            inputs.pop('loss_scale', None)
        return inputs, tokenizer_kwargs

    def _concat_context_list(
            self,
            context_list: List[Context],
            res_context_list: List[Context],  # inplace
            loss_scale_list: List[float],  # inplace
            system: Optional[str] = None,
            query: Optional[str] = None,
            response: Optional[str] = None,
            round0: Optional[int] = None,
            compute_loss: bool = True) -> None:
        """Concat context list and replace placeholder"""
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    if compute_loss:
                        content_part, weight_part = self.loss_scale(query, response)
                    else:
                        content_part, weight_part = [response], [0.]
                    res_context_list.extend(content_part)
                    loss_scale_list.extend(weight_part)
                    continue
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        context = context.replace(old_str, new_str)
            if len(context) == 0:
                continue
            res_context_list.append(context)
            loss_scale_list.append(0.)

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               **kwargs) -> Tuple[List[Context], List[float]]:
        """Merge anything in the context to simplify the inputs"""
        is_multi_modal: bool = kwargs.pop('is_multi_modal', False)

        if is_multi_modal:
            context_list, loss_scale_list = self.split_special_tokens(context_list, loss_scale_list)
        context_list, loss_scale_list = self.pre_tokenize(context_list, loss_scale_list, **kwargs)

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
    def split_special_tokens(context_list: List[Context],
                             loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        """Split special tokens, for example `<image>`, `<video>`, this will help the replace_tag operation"""
        from .utils import split_str_parts_by
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

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        """Override this function to do your own replace operation.

        This method is used to replace standard tags like `<image>` to some tokens that the model needs.

        Args:
            media_type: The modal.
            index: The index of the medias, for example 0 represents the first elements in `images`
            example: The input example

        Returns:
            The content or input_ids after replacement.
        """
        if media_type == 'image':
            return self.image_placeholder
        elif media_type == 'video':
            return ['<video>']
        elif media_type == 'audio':
            return ['<audio>']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        """Replace objects referenced by the bbox to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            index: The index in the `objects` key
            example: The input example

        Returns:
            The contents or input_ids replaced
        """
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [object_['caption']]
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            index: The index in the `objects` key
            example: The input example

        Returns:
            The contents or input_ids replaced
        """
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = ''
                for sub_object in object_['bbox']:
                    all_objects += f'[({sub_object[0]},{sub_object[1]}),' f'({sub_object[2]},{sub_object[3]})],'
                all_objects = all_objects[:-1]
                return [all_objects]
            else:
                return [f'[({object_["bbox"][0]},{object_["bbox"][1]}),({object_["bbox"][2]},{object_["bbox"][3]})]']
        else:
            return ['<bbox>']

    @classmethod
    def normalize_bbox(cls, objects: List[Dict[str, Any]], images: List[Any],
                       to_type: Literal['real', 'norm_1000', 'norm_1']) -> None:
        """Normalize bbox to needed.
        to_type support real/norm_1000/norm_1, which literally means the coordinates in real, or normalized by 1000,
            or normalized by 1.

        Args:
            objects: The objects containing the bbox
            images: The images list
            to_type: The coordinate type needed by the model.
        """
        if not objects or not images:
            return

        for object in objects:
            bbox = object['bbox']
            bbox_type = object['bbox_type']
            idx = object['image']
            image = images[idx]
            if bbox_type == 'real':
                if to_type == 'real':
                    continue
                width, height = image.width, image.height
                if isinstance(bbox[0], list):
                    bboxes = []
                    for _box in bbox:
                        bboxes.append([
                            int(coord / dim * 999) if to_type == 'norm_1000' else coord / dim
                            for coord, dim in zip(_box, [width, height, width, height])
                        ])
                    object['bbox'] = bboxes
                else:
                    object['bbox'] = [
                        int(coord / dim * 999) if to_type == 'norm_1000' else coord / dim
                        for coord, dim in zip(bbox, [width, height, width, height])
                    ]
                object['bbox_type'] = to_type
            elif bbox_type == 'norm_1000':
                if to_type == 'norm_1000':
                    continue
                if to_type == 'norm_1':
                    object['bbox'] = [coord / 999. for coord in bbox]
                elif to_type == 'real':
                    width, height = image.width, image.height
                    object['bbox'] = [
                        int(coord / 999. * dim) for coord, dim in zip(bbox, [width, height, width, height])
                    ]
                object['bbox_type'] = to_type
            elif bbox_type == 'norm_1':
                if to_type == 'norm_1':
                    continue
                if to_type == 'norm_1000':
                    object['bbox'] = [int(coord * 999) for coord in bbox]
                elif to_type == 'real':
                    width, height = image.width, image.height
                    object['bbox'] = [int(coord * dim) for coord, dim in zip(bbox, [width, height, width, height])]
                object['bbox_type'] = to_type

    def pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                     **kwargs) -> Tuple[List[Context], List[float]]:
        """This method happens before tokenization, replace standard tags to the contents or input_ids needed by
        the model.

        Args:
            context_list: The content list
            loss_scale_list: The loss scale list
        Returns:
            The context_list and loss_scale_list after replacement.
        """
        example = kwargs.get('example')  # get x_index
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        for k in ['image', 'video', 'audio']:
            example[f'{k}_index'] = 0

        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['image', 'video', 'audio']:
                if context == f'<{k}>':
                    c_list = self.replace_tag(k, example[f'{k}_index'], example)
                    example[f'{k}_index'] += 1
                    break
            else:
                if context == '<ref-object>':
                    c_list = self.replace_object(example.get('object_index', 0), example)
                    example['object_index'] = example.get('object_index', 0) + 1
                elif context == '<bbox>':
                    c_list = self.replace_box(example.get('box_index', 0), example)
                    example['box_index'] = example.get('box_index', 0) + 1
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

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
    def use_dynamic_eos(labels: List[int], suffix_tokens_id: List[int]) -> None:
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

    def _concat_and_tokenize(self,
                             messages: List[Dict[str, str]],
                             truncation_strategy: str,
                             auto_add_bos: bool = False,
                             **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        return: inputs, tokenizer_kwargs
        """
        system = [message for message in messages if message['role'] == 'system']
        messages = [message for message in messages if message['role'] != 'system']
        if len(system) > 0:
            system = system[0]['content']
        else:
            system = None

        assert len(messages) >= 1
        if len(messages) == 1:
            if messages['role'] == 'response':
                history = [None, messages['content']]
                history_roles = [None, messages['role']]
            else:
                history = [messages['content'], None]
                history_roles = [messages['role'], None]
        else:
            assert len(messages) % 2 == 0
            history = [[messages[i]['content'], messages[i+1]['content']] for i in range(len(messages) // 2)]
            history_roles = [[messages[i]['role'], messages[i + 1]['role']] for i in range(len(messages) // 2)]

        res_context_list: List[Context] = []
        loss_scale_list: List[float] = []
        if auto_add_bos:
            bos_token_id = self.tokenizer.bos_token_id
            if isinstance(bos_token_id, int) and bos_token_id in self.tokenizer.encode(''):
                res_context_list.append([bos_token_id])
                loss_scale_list.append(0.)
        prompt = self.prompt.copy()
        if system is None:
            prompt = [context for context in prompt if '{{SYSTEM}}' not in context]
        if system is None or any(['{{SYSTEM}}' in context for context in prompt]):
            prefix = self.prefix
        else:
            prefix = self.system_prefix
        self._concat_context_list(prefix, res_context_list, loss_scale_list, system=system)

        for i, ((q, r), (qr, rr)) in enumerate(zip(history, history_roles)):
            context_list = self.tool_prompt.copy() if qr == 'tool' else prompt.copy()
            extra_context_list = []
            is_suffix = False
            if i < len(history) - 1:
                context_list = [context for context in context_list if '{{SYSTEM}}' not in context]
                context_list.append('{{RESPONSE}}')
                if history[i + 1][0]:
                    extra_context_list = self.chat_sep
            elif r is not None:
                # last response
                context_list.append('{{RESPONSE}}')
                extra_context_list = self.suffix
                is_suffix = True
            if q or r:
                self._concat_context_list(
                    context_list,
                    res_context_list,
                    loss_scale_list,
                    query=q,
                    response=r,
                    system=system,
                    round0=i,
                    compute_loss=self.compute_per_round_loss or is_suffix)
                res_context_list += extra_context_list
                loss_scale_list += ([1.] if is_suffix else [0.]) * len(extra_context_list)
        inputs = {}
        if self.output_prompt_answer:
            # tokenizer_kwargs: use prompt
            answer_len = len(extra_context_list) + bool(history[-1][-1] is not None)
            total_len = len(res_context_list)
            for key, _slice in zip(['answer', 'prompt'],
                                   [slice(total_len - answer_len, total_len),
                                    slice(0, total_len - answer_len)]):
                _res_context_list, _loss_scale_list = self._simplify_context_list(res_context_list[_slice],
                                                                                  loss_scale_list[_slice], **kwargs)
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                    _res_context_list, _loss_scale_list)
                inputs[f'{key}_input_ids'], inputs[f'{key}_labels'] = input_ids, labels
                if self.loss_scale:
                    inputs[f'{key}_loss_scale'] = loss_scale
            input_ids = inputs['prompt_input_ids'] + inputs['answer_input_ids']
            labels = inputs['prompt_labels'] + inputs['answer_labels']
            if history[-1][-1] is None:
                assert len(inputs['answer_labels']) == 0
                inputs['answer_labels'] = None

        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, **kwargs)
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)
            if labels is not None:
                self.use_dynamic_eos(labels, self._encode_context_list(self.suffix)[0])

        if history[-1][-1] is None:
            labels = None

        if self.max_length is not None:
            if truncation_strategy == 'delete' and len(input_ids) > self.max_length:
                logger.warn(f'Current length of row({len(input_ids)}) is larger'
                            f' than the max_length({self.max_length}), deleted.')
                return {}, {}
            input_ids = input_ids[-self.max_length:]
            if labels is not None:
                labels = labels[-self.max_length:]
            if loss_scale is not None:
                loss_scale = loss_scale[-self.max_length:]
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels

        if self.loss_scale:
            inputs['loss_scale'] = loss_scale
        return inputs, tokenizer_kwargs

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        assert len(tokenizer_kwargs) == 0

    @staticmethod
    def pad_sequence(sequences: List[torch.Tensor],
                     padding_value: float = 0.,
                     padding_side: Literal['right', 'left'] = 'right') -> torch.Tensor:
        """Pad sequence by some side

        Args:
            sequences: The input sequences in tensor.
            padding_value: The padding value
            padding_side: The padding side

        Returns:
            A tensor after padding
        """
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.size(0) for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.size(0)
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        padding_right = self.padding_side == 'right'
        res = {}

        if 'inputs_embeds' in batch[0]:
            inputs_embeds = [b['inputs_embeds'] for b in batch]
            res['inputs_embeds'] = inputs_embeds
            res['attention_mask'] = [
                torch.ones((inputs_embeds[i].shape[0]), dtype=torch.int64) for i in range(len(inputs_embeds))
            ]
        elif 'input_ids' in batch[0]:
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            res['input_ids'] = input_ids
            res['attention_mask'] = [torch.ones(len(input_ids[i]), dtype=torch.int64) for i in range(len(input_ids))]

        for key in ['labels', 'loss_scale', 'position_ids']:
            if key in batch[0]:
                res[key] = [torch.tensor(b[key]) for b in batch]

        if padding_to is not None:
            assert 'input_ids' in res
            padding_len = padding_to - res['input_ids'][0].shape[-1]
            if padding_len > 0:
                for key, value in zip(['input_ids', 'attention_mask', 'labels', 'loss_scale', 'position_ids'],
                                      [tokenizer.pad_token_id, 0, -100, 0., -1]):
                    if key in res:
                        res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                            'constant', value)
        for key, value in zip(['input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'loss_scale', 'position_ids'],
                              [tokenizer.pad_token_id, 0., 0, -100, 0., -1]):
            if key in res:
                res[key] = self.pad_sequence(res[key], value, self.padding_side)

        if '_data' in batch[0]:
            res['_data'] = [b['_data'] for b in batch]
        # multimodal
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)
        return res

    @classmethod
    def get_generate_ids(cls, generate_ids: torch.Tensor, input_token_len: int) -> List[int]:
        if isinstance(generate_ids, torch.Tensor):
            generate_ids = generate_ids.tolist()
        if len(generate_ids) >= 1 and isinstance(generate_ids[0], (list, tuple)):
            generate_ids = generate_ids[0]
        return cls._get_generate_ids(generate_ids, input_token_len)

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids[input_token_len:]

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # copy from transformers.generation.streamers.TextStreamer
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    @classmethod
    def _get_safe_print_idx(cls, response: str, print_idx: int, is_finished: bool = False) -> int:
        if is_finished:
            return len(response)
        if response.endswith('\n') or len(response) > 0 and cls._is_chinese_char(ord(response[-1])):
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

    def post_process_generate_response(self, response: str, example: dict) -> str:
        return response
