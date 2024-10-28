# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import json
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase, StoppingCriteria

from swift.llm import History, Messages, to_device
from .agent import get_tools_prompt

Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'


class StopWordsCriteria(StoppingCriteria):
    """Adding extra stop words in template to prevent unstoppable generation
        Like suffixes and chat seps in the template.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: List[Word], **tokenizer_kwargs) -> None:
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


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], item_type: Optional[Type] = None) -> Any:
    if isinstance(element, (tuple, set, list)):
        for ele in element:
            out = fetch_one(ele)
            if out and (item_type is None or isinstance(out, item_type)):
                return out
    elif isinstance(element, dict):
        return fetch_one(list(element.values()))
    else:
        return element


def findall(token_list: List[int], sub_token_list: Union[int, List[int]]) -> List[int]:
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


def replace_img_tag(query: str,
                    response: Optional[str],
                    history: History,
                    replace_token: str,
                    pattern=r'<img>(.+?)</img>') -> Tuple[str, Optional[str], History, List[str]]:
    images_path = []
    new_history = []
    history = history.copy()
    history.append([query, response])
    for i, h in enumerate(history):
        new_h = []
        for content in h:
            if content is None:
                new_h.append(content)
            else:
                images_path += re.findall(pattern, content)
                new_h.append(re.sub(pattern, replace_token, content))
        new_history.append(new_h)
    return (*new_history[-1], new_history[:-1], images_path)


def align_image_inputs(input_ids: List[int], labels: List[int], new_input_ids,
                       image_token: int) -> Tuple[List[int], List[int]]:
    if isinstance(new_input_ids, torch.Tensor):
        new_input_ids = new_input_ids.tolist()

    # Find the tokens after the image_token in input_ids, and then align them.
    i, j = 0, 0
    while i < len(input_ids):
        x = input_ids[i]
        if x == image_token:
            assert i + 1 < len(input_ids), f'input_ids[-10:]: {input_ids[-10:]}'
            assert i - 1 >= 0, f'input_ids[:10]: {input_ids[:10]}'
            # [1, 2, 3(i-1), image_token(i), 4(i+1) ,5, 6]
            # [1, 2, 3(j_begin), a(j'), a, a, a, 4(j) ,5, 6]
            j_begin = j - 1
            for k in range(5):  # Increase robustness.
                if j_begin + k < len(new_input_ids) and new_input_ids[j_begin + k] == input_ids[i - 1]:
                    j_begin += k
                    break
                if j_begin - k >= 0 and new_input_ids[j_begin - k] == input_ids[i - 1]:
                    j_begin -= k
                    break
            else:
                raise ValueError(f'new_input_ids: {new_input_ids}, input_ids: {input_ids}')
            j_begin += 1
            while j < len(new_input_ids) and new_input_ids[j] != input_ids[i + 1]:
                j += 1
            input_ids = input_ids[:i] + new_input_ids[j_begin:j] + input_ids[i + 1:]
            if labels:
                labels = labels[:i] + [-100] * (j - j_begin) + labels[i + 1:]
            i += j - j_begin
        else:
            j += 1
        i += 1
    return input_ids, labels


@dataclass
class TemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    # None: use default system; '': not use system
    system: Optional[str] = None

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
        return self.__class__(
            deepcopy(self.messages), self.system, self.images.copy(), self.audios.copy(), self.videos.copy(),
            self.objects.copy())

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any], *, tools_prompt: str = 'react_en') -> 'TemplateInputs':
        messages = deepcopy(inputs['messages'])
        tools = deepcopy(inputs.get('tools'))
        objects = deepcopy(inputs.get('objects')) or []

        assert len(messages) >= 1

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

        media_kwargs = TemplateInputs.remove_messages_media(messages)
        for k in list(media_kwargs.keys()):
            mm_data = media_kwargs[k]
            inputs_mm_data = inputs.get(k).copy() or []
            if mm_data:
                assert not inputs_mm_data, f'self.{k}: {inputs_mm_data}'
            else:
                media_kwargs[k] = inputs_mm_data

        TemplateInputs.messages_join_observation(messages)
        return cls(messages, system, **media_kwargs, objects=objects)

    @staticmethod
    def remove_messages_media(messages: Messages) -> Dict[str, Any]:
        res = {'images': [], 'audios': [], 'videos': []}
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                continue
            # List[Dict[str, Any]]
            new_content = ''
            for item in content:
                key: str = item['type']
                value = item[key]
                if key == 'text':
                    new_content += value
                    continue
                # image/audio/video
                # image_url/audio_url/video_url
                if key.endswith('_url'):
                    key = key[:-len('_url')]
                new_content += f'<{key}>'
                res[f'{key}s'].append(value)
            message['content'] = new_content
        return res

    @classmethod
    def from_infer_request(cls, infer_request: 'InferRequest', *, tools_prompt: str = 'react_en') -> 'TemplateInputs':
        return cls.from_dict(asdict(infer_request), tools_prompt=tools_prompt)

    @staticmethod
    def messages_join_observation(messages: Messages) -> None:
        """
        Joins observations from 'tool' message into the 'assistant' response.

        Example:
        ---------
        Original messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                    [{"location": "Hangzhou"}]\nObservations:'},
            {'role': 'tool', 'content': 'It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]

        Transformed messages:
        messages = [
            {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
            {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                    [{"location": "Hangzhou"}]\nObservations: It is 26 degrees Celsius and sunny in Hangzhou today.'}
        ]
        """
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool' and isinstance(content,
                                                                         str) and content.endswith('Observation:'):
                assert isinstance(pre_content, str)
                pre_message['content'] = pre_content + content  # assistant
                messages.pop(i)  # remove tool
            else:
                i += 1
