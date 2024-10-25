# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from functools import partial, wraps
from types import MethodType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import json
import torch
import torch.nn.functional as F
import transformers
from packaging import version
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, StoppingCriteria
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import strtobool

from swift.llm.agent.utils import calculate_loss_scale, get_tools_prompt
from swift.torchacc_utils import pad_and_split_batch
from swift.utils import get_dist_setting, get_logger, upper_bound, use_torchacc
from .vision_utils import (load_audio_qwen, load_batch, load_image, load_video_cogvlm2, load_video_internvl,
                           load_video_llava, load_video_minicpmv_mplug_owl3, load_video_qwen2, rescale_image,
                           transform_image)

logger = get_logger()

DEFAULT_SYSTEM = 'You are a helpful assistant.'
History = List[Union[Tuple[str, str], List[str]]]
Prompt = List[Union[str, List[int], List[str]]]
StopWords = Prompt
Context = Union[str, List[int]]
TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    chatglm_generation = 'chatglm-generation'
    qwen_vl_generation = 'qwen-vl-generation'
    qwen_audio_generation = 'qwen-audio-generation'
    # chat
    default = 'default'
    qwen = 'qwen'
    qwen2_5 = 'qwen2_5'
    qwen_vl = 'qwen-vl'
    qwen_audio = 'qwen-audio'
    qwen2_audio = 'qwen2-audio'
    qwen2_audio_generation = 'qwen2-audio-generation'
    qwen2_vl = 'qwen2-vl'
    qwen2_vl_generation = 'qwen2-vl-generation'
    modelscope_agent = 'modelscope-agent'
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    chatglm4 = 'chatglm4'
    codegeex4 = 'codegeex4'
    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llama3_1_omni = 'llama3_1-omni'
    llama3_2 = 'llama3_2'
    llama3_2_vision = 'llama3_2-vision'
    llama3_2_vision_generation = 'llama3_2-vision-generation'
    reflection = 'reflection'
    longwriter_llama3 = 'longwriter-llama3'
    # llava-hf
    llava1_5 = 'llava1_5'
    llava_mistral = 'llava-mistral'
    llava_vicuna = 'llava-vicuna'
    llava_yi = 'llava-yi'
    llama3_llava_next_hf = 'llama-llava-next-hf'
    llava_next_llama3 = 'llava-next-llama3'
    llava_qwen_hf = 'llama-qwen-hf'
    llava_onevision_qwen = 'llava-onevision-qwen'
    # llava-video
    llava_next_video = 'llava-next-video'
    llava_next_video_yi = 'llava-next-video-yi'
    # lmms-lab:llava
    llama3_llava_next = 'llama3-llava-next'
    llava_qwen = 'llava-qwen'
    # xtuner:llava
    llava_llama_instruct = 'llava-llama-instruct'

    idefics3 = 'idefics3'
    mistral_nemo = 'mistral-nemo'
    pixtral = 'pixtral'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm_xcomposer2 = 'internlm-xcomposer2'
    internlm_xcomposer2_4khd = 'internlm-xcomposer2-4khd'
    internlm_xcomposer2_5 = 'internlm-xcomposer2_5'
    internvl = 'internvl'
    internvl2 = 'internvl2'
    internvl_phi3 = 'internvl-phi3'
    internvl2_phi3 = 'internvl2-phi3'
    florence = 'florence'
    yi_coder = 'yi-coder'
    yi_vl = 'yi-vl'
    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    deepseek = 'deepseek'
    numina_math = 'numina-math'
    deepseek_coder = 'deepseek-coder'
    deepseek_vl = 'deepseek-vl'
    deepseek2 = 'deepseek2'
    deepseek2_5 = 'deepseek2_5'
    codefuse_codellama = 'codefuse-codellama'
    codefuse = 'codefuse'
    cogvlm = 'cogvlm'
    cogvlm2_video = 'cogvlm2-video'
    glm4v = 'glm4v'
    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    orion = 'orion'
    minicpm = 'minicpm'
    minicpm_v = 'minicpm-v'
    minicpm_v_v2_5 = 'minicpm-v-v2_5'
    minicpm_v_v2_6 = 'minicpm-v-v2_6'
    gemma = 'gemma'
    paligemma = 'paligemma'
    mplug_owl2 = 'mplug-owl2'
    mplug_owl3 = 'mplug_owl3'
    wizardlm2_awq = 'wizardlm2-awq'
    wizardlm2 = 'wizardlm2'
    atom = 'atom'
    phi3 = 'phi3'
    phi3_vl = 'phi3-vl'
    telechat = 'telechat'
    telechat2 = 'telechat2'
    dbrx = 'dbrx'
    mengzi = 'mengzi'
    c4ai = 'c4ai'
    aya = 'aya'
    chatml = 'chatml'
    got_ocr2 = 'got_ocr2'
    ovis1_6 = 'ovis1_6'
    molmo = 'molmo'
    deepseek_janus = 'deepseek-janus'
    emu3_chat = 'emu3-chat'
    # compatibility. (Deprecated)
    default_generation_bos = 'default-generation-bos'
    yi = 'yi'
    yi1_5 = 'yi1_5'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


class StopWordsCriteria(StoppingCriteria):
    # The returned sentence includes stop words.
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


def is_deepspeed_enabled():
    return strtobool(os.environ.get('ACCELERATE_USE_DEEPSPEED', 'False'))


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

        Examples:
            <start>system\nYou are a helpful assistant!<end>\n<bos><start>Who are you?<end>\n<start>assistant:I am a robot<end>\n<start>Who are you?<end>\n<start>assistant:I am a robot<end> # noqa
            --------------- --------------------------         ---  ----- ------------ ----------------------- ----------- ----                                                         -----
             system_prefix          system                   prefix prompt   query              prompt           response chat_sep                                                      suffix
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
                 padding_side: Literal['left', 'right'] = 'right') -> None:
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
        self._is_vllm = False
        self._is_lmdeploy = False
        self._is_training = False
        self.padding_side = padding_side

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        return [p.replace('{{SYSTEM}}', '') for p in prefix if '{{SYSTEM}}' in p]

    @staticmethod
    def _has_system(prefix: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix])

    @staticmethod
    def _preprocess_prompt(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
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

    def _init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
                       model: torch.nn.Module = None,
                       **kwargs) -> None:
        assert self._is_init is False, 'The template has been initialized.'
        self.is_multimodal = getattr(tokenizer, 'is_multimodal', None)
        self._is_init = True
        self.tokenizer = tokenizer
        # if default_system is None. not change self.default_system
        if default_system == '':
            self.default_system = None
        elif default_system is not None:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {getattr(self, "template_type", None)}')
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.model = model
        self.ref_model = kwargs.get('ref_model', None)
        self.use_loss_scale = kwargs.get('use_loss_scale', False)
        self.response_loss_scale_map = kwargs.get('loss_scale_map', None)
        self.query_loss_scale_map = None
        if self.response_loss_scale_map is not None:
            if 'query' in self.response_loss_scale_map and isinstance(self.response_loss_scale_map['query'], dict):
                self.query_loss_scale_map = self.response_loss_scale_map['query']
            if 'response' in self.response_loss_scale_map and isinstance(self.response_loss_scale_map['response'],
                                                                         dict):
                self.response_loss_scale_map = self.response_loss_scale_map['response']

        self.sequence_parallel_size = kwargs.get('sequence_parallel_size', 1)
        self.rescale_image = kwargs.get('rescale_image', -1)

        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self._preprocess_prompt(tokenizer, value)
            setattr(self, key, value)

    @contextmanager
    def training_context(self):
        if self.model is None:
            self._is_training = True
            yield
            self._is_training = False
            return

        self._is_training = True

        def _pre_forward_hook(module, args, kwargs):
            from .utils import to_device
            if '_data' in kwargs:
                res_extra = []
                data = kwargs.pop('_data')
                for d in data:
                    res_extra.append(self._post_encode(module, d))
                kwargs.update(to_device(self.data_collator(res_extra), module.device))
                if 'inputs_embeds' in kwargs:
                    kwargs.pop('input_ids', None)

            if isinstance(module, PeftModel):
                parameters = inspect.signature(module.base_model.model.forward).parameters
            else:
                parameters = inspect.signature(module.forward).parameters

            if 'position_ids' not in parameters:
                kwargs.pop('position_ids', None)
            return args, kwargs

        parameters = inspect.signature(self.model.register_forward_pre_hook).parameters
        handle, handle2 = None, None
        deepspeed = None
        if 'with_kwargs' in parameters:
            handle = self.model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)
            if self.ref_model:
                handle2 = self.ref_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)
            if is_deepspeed_zero3_enabled():
                import deepspeed
                _old_initialize = deepspeed.initialize

                @wraps(_old_initialize)
                def _initialize(*args, **kwargs):
                    res = _old_initialize(*args, **kwargs)
                    self.model._forward_pre_hooks.move_to_end(handle.id)
                    if self.ref_model:
                        self.ref_model._forward_pre_hooks.move_to_end(handle2.id)
                    return res

                deepspeed.initialize = _initialize
        yield
        self._is_training = False
        if handle:
            handle.remove()
        if handle2:
            handle2.remove()
        if deepspeed:
            deepspeed.initialize = _old_initialize

    @contextmanager
    def vllm_context(self):
        self._is_vllm = True
        yield
        self._is_vllm = False

    @contextmanager
    def lmdeploy_context(self):
        self._is_lmdeploy = True
        yield
        self._is_lmdeploy = False

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        return {}

    def check_example(self, example: Dict[str, Any]) -> None:
        pass

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        history: History = deepcopy(example.get('history') or [])
        query: str = example.get('query') or ''
        response: str = example.get('response') or ''
        history.append([query, response])
        for media_key, media_tag in [('videos', '<video>'), ('images', '<image>'), ('audios', '<audio>')]:
            if example.get(media_key):
                infer_media_type = TEMPLATE_MAPPING[self.template_type].get('infer_media_type')
                if infer_media_type == 'round':
                    n_round = len(example[media_key])
                    assert n_round == len(history)
                    for i, h, m in zip(range(n_round), history, example[media_key]):
                        content = f'{h[0]}\n{h[1]}'
                        num_media_tags = len(re.findall(media_tag, content))
                        if m:
                            assert num_media_tags <= 1, (
                                'The model includes at most one media per round. However, '
                                f'this round contains {num_media_tags} media_tags. query: {h[0]}, response: {h[1]}')
                            if num_media_tags == 0:
                                h[0] = media_tag + h[0]
                        else:
                            assert num_media_tags == 0, f'Missing media. query: {h[0]}'
                        history[i][0] = h[0]

                    example[media_key] = [m for m in example[media_key] if m]

                else:
                    num_media_tags = len(re.findall(media_tag, '\n'.join([f'{h[0]}\n{h[1]}' for h in history])))
                    example[media_key] = [m for m in example[media_key] if m]
                    num_media = len(example[media_key])
                    num_new_tags = num_media - num_media_tags
                    assert num_new_tags >= 0, f'Number of media: {num_media}, number of media_tags: {num_media_tags}'
                    history[0][0] = media_tag * num_new_tags + history[0][0]
        example['query'] = history[-1][0]
        if example.get('response') is not None:
            example['response'] = history[-1][1]
        example['history'] = history[:-1]

    def replace_media_tags(self, example) -> None:
        if self.is_multimodal in {True, None}:
            for k, tag, pattern in zip(['images', 'audios', 'videos'], ['<image>', '<audio>', '<video>'],
                                       [r'<img>(.+?)</img>', r'<audio>(.+?)</audio>', r'<video>(.+?)</video>']):
                example['query'], example['response'], example['history'], medias_path = replace_img_tag(
                    example.get('query'), example.get('response'),
                    example.get('history') or [], tag, pattern)
                if example.get(k) and medias_path:
                    raise ValueError(f'Do not mix use the {pattern} tag and {tag} tag.')
                example[k] = example.get(k) or [] + medias_path

    def _preprocess_media(self, example):
        from .media import MediaTag
        from .client_utils import decode_base64
        # Format media_keys to list
        for media_key in MediaTag.media_keys.values():
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
            if example.get('objects') or self.load_medias or self._is_lmdeploy or self._is_vllm:
                images = load_batch(images, load_image)  # base64/local_path -> PIL.Image
            if example.get('objects'):
                # Normalize grounding bboxes
                self.normalize_bbox(example['objects'], images, to_type=self.grounding_type)
            if self.load_medias and self.grounding_type != 'real':
                images = [rescale_image(img, self.rescale_image) for img in images]
            if not self.load_medias and not self._is_lmdeploy and not self._is_vllm:  # fix pt & qwen-vl
                images = decode_base64(images=images)['images']  # PIL.Image/base64 -> local_path
            example['images'] = images

        # Check the example that whether matching the very template's rules
        self.check_example(example)

    def preprocess(self, example):
        # Duplicate example and create a new one to prepare in-place changes
        example = example.copy()
        template_type: Optional[str] = getattr(self, 'template_type', None)
        tools: Union[List[Any], str] = example.get('tools') or []

        # Template needs to be initialized
        if not self._is_init:
            raise ValueError(
                'Template is not initialized, please use the `get_template` function to obtain the template.')

        # Reset system (by default value and agent tools)
        system: Optional[str] = example.get('system', None)
        if system is None:
            if self.use_default_system:
                system = self.default_system
        elif system == '':
            system = None
        else:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {template_type}')
        if tools:
            if isinstance(tools, str):
                tools = json.loads(tools)
            if system is None:
                system = ''
            system += get_tools_prompt(tools, self.tools_prompt)

        example['system'] = system

        # Check whether this template supports multi-round
        history: History = example.get('history') or []
        if len(history) > 0:
            assert self.support_multi_round, (
                f'The template does not support multi-round chat, template_type: {template_type}')

        # Set history_roles
        history_roles: Optional[History] = example.get('history_roles')
        if history_roles is None:
            example['history_roles'] = [['user', 'assistant'] for _ in range(len(history))]

        self._preprocess_media(example)
        return example

    def encode(self, example: Dict[str, Any], streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from .utils import to_device
        example = self.preprocess(example)
        _encode = self._encode
        if self._is_lmdeploy or self._is_vllm:
            assert self.is_multimodal is not None, 'Please use the get_model_tokenizer function.'
            _encode = MethodType(Template._encode, self)
        res = _encode(example)
        inputs = res[0]
        if not self._is_training and '_data' in inputs:
            data = inputs.pop('_data')
            data = to_device(data, self.model.device)
            inputs.update(self._post_encode(self.model, data))
        return res if not streaming else inputs

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        ranges = []
        for i in range(len(idx_list) - 1):
            _range = []
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            _range.append(len(new_input_ids))
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            _range.append(len(new_input_ids))
            ranges.append(_range)
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_embeddings'] = images
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """return: inputs, tokenizer_kwargs"""
        query: str = example.get('query') or ''
        query_role: str = example.get('query_role') or 'user'
        response: Optional[str] = example.get('response')
        history: History = example.get('history') or []
        history_roles: Optional[History] = example.get('history_roles')
        system: Optional[str] = example.get('system', None)
        is_multi_modal: bool = any([example.get(key) for key in Template.special_keys])

        inputs, tokenizer_kwargs = self._concat_and_tokenize(
            query,
            query_role,
            response,
            history,
            history_roles,
            system,
            self.truncation_strategy,
            auto_add_bos=self.auto_add_bos,
            example=example,
            is_multi_modal=is_multi_modal)
        if self._is_lmdeploy or self._is_vllm:
            for key in ['images', 'audios', 'videos']:
                inputs[key] = example.get(key)
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
        # concat context list and replace placeholder
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    if compute_loss:
                        content_part, weight_part = calculate_loss_scale(query, response, self.use_loss_scale,
                                                                         self.response_loss_scale_map,
                                                                         self.query_loss_scale_map)
                    else:
                        content_part, weight_part = [response], [0.]
                    res_context_list.extend(content_part)
                    loss_scale_list.extend(weight_part)
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
            loss_scale_list.append(0.)

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               **kwargs) -> Tuple[List[Context], List[float]]:
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
        from swift.utils.utils import split_str_parts_by
        res: List[Context] = []
        loss_scale_res: List[float] = []
        from .utils import fetch_one
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
        if media_type == 'image':
            if self._is_lmdeploy:
                return [[-100]]
            else:
                return self.image_placeholder
        elif media_type == 'video':
            return ['<video>']
        elif media_type == 'audio':
            return ['<audio>']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [object_['caption']]
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = ''
                for sub_object in object_['bbox']:
                    all_objects += (f'[({sub_object[0]},{sub_object[1]}),' f'({sub_object[2]},{sub_object[3]})],')
                all_objects = all_objects[:-1]
                return [all_objects]
            else:
                return [f'[({object_["bbox"][0]},{object_["bbox"][1]}),({object_["bbox"][2]},{object_["bbox"][3]})]']
        else:
            return ['<bbox>']

    @classmethod
    def normalize_bbox(cls, objects, images, to_type: Literal['real', 'norm_1000', 'norm_1']):
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
        # replace tag/object/box
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
                    loss_scale = 0.
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
                             query: str,
                             query_role: str,
                             response: Optional[str],
                             history: History,
                             history_roles: History,
                             system: Optional[str],
                             truncation_strategy: str,
                             auto_add_bos: bool = False,
                             **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        return: inputs, tokenizer_kwargs
        """
        history = history.copy()
        history_roles = history_roles.copy()

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

        history.append([query, response])
        history_roles.append([query_role, 'assistant'])

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
            answer_len = len(extra_context_list) + bool(response is not None)
            total_len = len(res_context_list)
            for key, _slice in zip(['answer', 'prompt'],
                                   [slice(total_len - answer_len, total_len),
                                    slice(0, total_len - answer_len)]):
                _res_context_list, _loss_scale_list = self._simplify_context_list(res_context_list[_slice],
                                                                                  loss_scale_list[_slice], **kwargs)
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                    _res_context_list, _loss_scale_list)
                inputs[f'{key}_input_ids'], inputs[f'{key}_labels'] = input_ids, labels
                if self.use_loss_scale:
                    inputs[f'{key}_loss_scale'] = loss_scale
            input_ids = inputs['prompt_input_ids'] + inputs['answer_input_ids']
            labels = inputs['prompt_labels'] + inputs['answer_labels']
            if response is None:
                assert len(inputs['answer_labels']) == 0
                inputs['answer_labels'] = None

        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, **kwargs)
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)
            if labels is not None:
                self.use_dynamic_eos(labels, self._encode_context_list(self.suffix)[0])

        if response is None:
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

        if self.use_loss_scale:
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
                     padding_side: Literal['right', 'left'] = 'right'):
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

        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if use_torchacc():
            rank, _, world_size, _ = get_dist_setting()
            input_ids, attention_mask, labels, loss_scale = pad_and_split_batch(
                padding_to,
                input_ids,
                attention_mask,
                labels,
                loss_scale,
                self.max_length,
                self.tokenizer,
                rank,
                world_size,
                padding_right=padding_right)
        if self.sequence_parallel_size > 1 and input_ids is not None:
            bs, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
            assert padding_right or bs == 1, 'Sequence parallel only support padding_side=right'
            from swift.trainers.xtuner import get_xtuner_sequence_parallel_world_size
            if get_xtuner_sequence_parallel_world_size() > 1:
                from swift.trainers.xtuner import pad_and_split_for_sequence_parallel
                input_ids, labels, position_ids, attention_mask, loss_scale = \
                    pad_and_split_for_sequence_parallel(
                        tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale)
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
            if value is not None:
                res[key] = value

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
        elif isinstance(generate_ids, tuple):
            generate_ids = list(generate_ids)
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


def register_template(template_type: str, template: Template, *, exist_ok: bool = False, **kwargs) -> None:
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    template.template_type = template_type
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


register_template(
    TemplateType.default,
    Template([], ['### Human:\n{{QUERY}}\n\n### Assistant:\n'], ['\n\n'], [['eos_token_id']],
             DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}'], None, [['eos_token_id']], auto_add_bos=True)


register_template(TemplateType.default_generation, DefaultGenerationTemplate(), is_generation=True)
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]),
    is_generation=True)


class ChatmlTemplateMixin:
    system = None

    def __init__(self, auto_add_bos: bool = True):
        Template.__init__(
            self, [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
            ['<|im_end|>'],
            self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            auto_add_bos=auto_add_bos)


class ChatmlTemplate(ChatmlTemplateMixin, Template):
    pass


class QwenTemplateMixin(ChatmlTemplateMixin):
    system = DEFAULT_SYSTEM

    def __init__(self):
        super().__init__(auto_add_bos=False)


class QwenTemplate(QwenTemplateMixin, Template):
    pass


class GOTImageEvalProcessor:

    def __init__(self, image_size=384, mean=None, std=None):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, item):
        return self.transform(item)


class GOT_OCR2Template(QwenTemplate):
    system = '        You should follow the instructions carefully and explain your answers in detail.'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        # OCR:
        # OCR with format:
        assert media_type == 'image'
        return ['<img>' + '<imgpad>' * 256 + '</img>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        image_processor_high = GOTImageEvalProcessor(image_size=1024)
        for i, image in enumerate(images):
            images[i] = image_processor_high(image)[None].to(self.model.dtype)
        if images:
            inputs['images'] = images
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = _gather_list(batch, 'images')
        if images:
            res['images'] = images
        return res


register_template(TemplateType.got_ocr2, GOT_OCR2Template(), lazy_tokenize=True, use_model=True)


class OVIS1_6Template(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, [-200])
        added_tokens_len = 0
        pixel_values = []
        for i, idx in enumerate(idx_list):
            max_partition = get_env_args('max_partition', int, 9)
            raw_pixel_values, image_placeholders = self.model.visual_tokenizer.preprocess_image(
                images[i], max_partition=max_partition)
            input_ids = input_ids[:idx] + image_placeholders + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(image_placeholders) + labels[idx + 1:]
            pixel_values.append(raw_pixel_values)
            added_tokens_len += len(image_placeholders) - 1
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0).to(self.model.visual_tokenizer.dtype)
        else:
            pixel_values = None
        inputs = {'labels': labels}
        if labels is not None:
            labels = torch.tensor(labels)[None]
        inputs['_data'] = {'input_ids': torch.tensor(input_ids)[None], 'labels': labels, 'pixel_values': [pixel_values]}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        _, inputs_embeds, labels, _ = self.model.merge_multimodal(
            text_input_ids=data['input_ids'],
            text_attention_masks=torch.ones_like(data['input_ids']),  # not use, only compat
            text_labels=data['labels'],
            pixel_values=data['pixel_values'],
            left_padding=True)
        return {'inputs_embeds': inputs_embeds[0], 'labels': labels}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(
    TemplateType.ovis1_6,
    OVIS1_6Template(['<bos>'], ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
                    ['<end_of_turn>\n'], ['<end_of_turn>'], None,
                    ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n']),
    lazy_tokenize=True,
    use_model=True)


class _QwenVLTemplateMixin:
    load_medias = False

    def check_example(self, example):
        if self._is_lmdeploy or self._is_vllm:
            return
        images = example.get('images') or []
        from .utils import fetch_one
        assert not images or isinstance(fetch_one(images), str), 'QwenVL only supports datasets with images paths!'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'image'
        if self._is_lmdeploy:
            return [f'Picture {index + 1}: ', [-100], '\n']
        else:
            images = example.get('images') or []
            image = images[index]
            if self._is_vllm:
                return [f'Picture {index + 1}: <img></img>\n']
            else:
                assert isinstance(image, str)
                return [f'Picture {index + 1}: <img>{image}</img>\n']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        return [f'<ref>{object_["caption"]}</ref>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                all_objects += (f'<box>({sub_object[0]},{sub_object[1]}),' f'({sub_object[2]},{sub_object[3]})</box>')
            return [all_objects]
        else:
            return [
                f'<box>({object_["bbox"][0]},{object_["bbox"][1]}),'
                f'({object_["bbox"][2]},{object_["bbox"][3]})</box>'
            ]


class Qwen2_5Template(QwenTemplate):
    system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'


register_template(TemplateType.qwen, QwenTemplate())
register_template(TemplateType.qwen2_5, Qwen2_5Template())


class QwenVLTemplate(_QwenVLTemplateMixin, QwenTemplate):
    pass


class QwenVLGenerationTemplate(_QwenVLTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.qwen_vl, QwenVLTemplate())
register_template(TemplateType.qwen_vl_generation, QwenVLGenerationTemplate())

register_template(TemplateType.chatml, ChatmlTemplate())
register_template(TemplateType.yi, ChatmlTemplate())
register_template(TemplateType.yi1_5, ChatmlTemplate())

register_template(
    TemplateType.modelscope_agent,
    Template([], [' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'], [], [' \n\n</s>'], DEFAULT_SYSTEM,
             [' \n\n<|system|>:{{SYSTEM}}']))


class _QwenAudioTemplateMixin:

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        audios = example.get('audios') or []
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, tokenizer_kwargs
        inputs.pop('loss_scale', None)
        inputs.update(tokenizer_kwargs)
        return inputs, tokenizer_kwargs

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        return {'audio_info': self.tokenizer.process_audio(context)}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        audio_info = curr_tokenizer_kwargs.get('audio_info')
        old_audio_info = tokenizer_kwargs.get('audio_info')
        if old_audio_info is None:
            tokenizer_kwargs['audio_info'] = audio_info
        elif audio_info is not None:
            for k in ['input_audios', 'input_audio_lengths']:
                old_audio_info[k] = torch.concat([old_audio_info[k], audio_info[k]], dim=0)
            for k in ['audio_span_tokens', 'audio_urls']:
                old_audio_info[k] = old_audio_info[k] + audio_info[k]

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = Template.data_collator(self, batch, padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


class QwenAudioTemplate(_QwenAudioTemplateMixin, QwenTemplate):
    pass


class QwenAudioGenerationTemplate(_QwenAudioTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.qwen_audio, QwenAudioTemplate(), lazy_tokenize=True)
register_template(
    TemplateType.qwen_audio_generation, QwenAudioGenerationTemplate(), lazy_tokenize=True, is_generation=True)


class _Qwen2AudioTemplateMixin:

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        sampling_rate = processor.feature_extractor.sampling_rate
        audios = load_batch(
            example.get('audios') or [], load_func=partial(load_audio_qwen, sampling_rate=sampling_rate))
        if audios:
            audio_inputs = processor.feature_extractor(
                audios, sampling_rate=sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            inputs.update(audio_inputs)
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = Template.data_collator(self, batch, padding_to)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


class Qwen2AudioTemplate(_Qwen2AudioTemplateMixin, QwenTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        return [f'Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>\n']


class Qwen2AudioGenerationTemplate(_Qwen2AudioTemplateMixin, DefaultGenerationTemplate):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        return ['<|audio_bos|><|AUDIO|><|audio_eos|>\n']


register_template(TemplateType.qwen2_audio, Qwen2AudioTemplate(), lazy_tokenize=True)

register_template(
    TemplateType.qwen2_audio_generation, Qwen2AudioGenerationTemplate(), lazy_tokenize=True, is_generation=True)


def _process_image_qwen(image):
    from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, smart_resize
    size_factor = get_env_args('image_factor', int, IMAGE_FACTOR, ['size_factor'])
    # resize
    resized_height = get_env_args('resized_height', int, None)
    resized_width = get_env_args('resized_width', int, None)
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = get_env_args('min_pixels', int, MIN_PIXELS)
        max_pixels = get_env_args('max_pixels', int, MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


class _Qwen2VLTemplateMixin:
    image_token_id = 151655
    video_token_id = 151656

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type in {'image', 'video'}
        if media_type == 'image':
            example['images'][index] = _process_image_qwen(example['images'][index])
            return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            example['videos'][index] = load_video_qwen2(example['videos'][index])
            return ['<|vision_start|><|video_pad|><|vision_end|>']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return ['<|object_ref_start|>', object_['caption'], '<|object_ref_end|>']
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = ''
                for sub_object in object_['bbox']:
                    all_objects += (f'<|box_start|>({sub_object[0]},{sub_object[1]}),'
                                    f'({sub_object[2]},{sub_object[3]})<|box_end|>')
                return [all_objects]
            else:
                return [
                    f'<|box_start|>({object_["bbox"][0]},{object_["bbox"][1]}),'
                    f'({object_["bbox"][2]},{object_["bbox"][3]})<|box_end|>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        images = example.get('images') or []
        videos = example.get('videos') or []
        data = {}
        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(images=None, videos=videos, return_tensors='pt')
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                idx_list = _findall(input_ids, media_token)
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    merge_length = processor.image_processor.merge_size**2
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    input_ids = input_ids[:idx
                                          + added_tokens_len] + [media_token] * token_len + input_ids[added_tokens_len
                                                                                                      + idx + 1:]
                    if labels:
                        labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx
                                                                                               + 1:]
                    added_tokens_len += token_len - 1
                data.update(media_inputs)

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        data['input_ids'] = torch.tensor(input_ids)[None]
        inputs['_data'] = data
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        input_ids = data['input_ids']
        pixel_values = data.get('pixel_values')
        pixel_values_videos = data.get('pixel_values_videos')
        inputs_embeds = _model.embed_tokens(input_ids)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                processor = self.tokenizer.processor
                media_inputs = processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                pixel_values = media_inputs['pixel_values'].to(device)

                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                image_grid_thw = data['image_grid_thw']
                pixel_values = pixel_values.type(model.visual.get_dtype())
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_grid_thw = data['video_grid_thw']
                pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return {'inputs_embeds': inputs_embeds[0]}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        for media_type in ['image', 'video']:
            grid_thw = [b[f'{media_type}_grid_thw'] for b in batch if b.get(f'{media_type}_grid_thw') is not None]
            if grid_thw:
                res[f'{media_type}_grid_thw'] = torch.concat(grid_thw)
        if 'input_ids' in res:
            # fix https://github.com/huggingface/transformers/pull/33487
            position_ids, _ = self.model.get_rope_index(res['input_ids'], res.get('image_grid_thw'),
                                                        res.get('video_grid_thw'), res['attention_mask'])
            res['position_ids'] = position_ids.contiguous()
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


class Qwen2VLTemplate(_Qwen2VLTemplateMixin, QwenTemplate):
    pass


class Qwen2VLGenerationTemplate(_Qwen2VLTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.qwen2_vl, Qwen2VLTemplate(), lazy_tokenize=True)

register_template(TemplateType.qwen2_vl_generation, Qwen2VLGenerationTemplate(), lazy_tokenize=True, is_generation=True)


def _gather_list(batch: List[Dict[str, Any]], attr_name: str) -> Optional[List[Any]]:
    # List[Tensor] ->  List[Tensor]
    res = []
    for b in batch:
        if b.get(attr_name) is not None:
            res += b.pop(attr_name)
    return res


class PixtralTemplate(Template):

    def __init__(self):
        super().__init__(['<s>{{SYSTEM}}'], ['[INST]{{QUERY}}[/INST]'], ['</s>'], ['</s>'], None)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        return ['[IMG]']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        processor = self.tokenizer.processor
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 10)
        if idx_list:
            image_inputs = processor.image_processor(images, patch_size=processor.patch_size, return_tensors='pt')
            inputs['pixel_values'] = image_inputs['pixel_values'][0]
            image_sizes = image_inputs['image_sizes'][0]
            added_tokens_len = 0
            for idx, image_size in zip(idx_list, image_sizes):
                height, width = image_size
                num_height_tokens = height // processor.patch_size
                num_width_tokens = width // processor.patch_size
                replace_tokens = [processor.image_token * num_width_tokens + processor.image_break_token] * (
                    num_height_tokens - 1)
                replace_tokens += [processor.image_token * num_width_tokens + processor.image_end_token]
                # Flatten list
                replace_str = ''.join(replace_tokens)
                img_tokens: List[int] = self.tokenizer.encode(replace_str, add_special_tokens=False)
                input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
                if labels is not None:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                                 + 1:]
                added_tokens_len += len(img_tokens) - 1
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels

        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        pixel_values = _gather_list(batch, 'pixel_values')
        res = super().data_collator(batch, padding_to)
        if pixel_values:
            res['pixel_values'] = pixel_values
        return res


register_template(TemplateType.pixtral, PixtralTemplate(), lazy_tokenize=True)


class YiCoderTemplate(ChatmlTemplate):
    system = 'You are a helpful assistant.'


register_template(TemplateType.yi_coder, YiCoderTemplate())

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


class YiVLTemplate(Template):

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            inputs['images'] = image_tensor.to(model.dtype)
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'YIVL does not support mix-batch nlp dataset and multi-modal dataset'
        return res


class GLMTemplate(Template):

    def _init_template(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        res = super()._init_template(tokenizer, *args, **kwargs)
        token_list = tokenizer.encode('')
        self.prefix.insert(0, token_list)
        if self.system_prefix is not None:
            self.system_prefix.insert(0, token_list)
        return res


class GLM4VTemplate(GLMTemplate):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|assistant|>'], [], ['<|endoftext|>'], None,
                         ['<|system|>\n{{SYSTEM}}'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from .utils import history_to_messages

        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = example.get('images')[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            messages = history_to_messages(example.get('history') or [], example['query'], example.get('system'))
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.tokenizer.apply_chat_template(messages, return_dict=True)
            inputs['images'] = inputs2['images']
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(TemplateType.glm4v, GLM4VTemplate(), infer_media_type='dialogue', lazy_tokenize=True, use_model=True)

register_template(
    TemplateType.yi_vl,
    YiVLTemplate([], [[8308], 'Human: {{QUERY}}\n', [8308], 'Assistant:'], ['\n'], ['\n', [8308]], yi_vl_default_system,
                 ['{{SYSTEM}}\n\n']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)

register_template(TemplateType.baichuan, Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [], [['eos_token_id']]))

register_template(
    TemplateType.chatglm2,
    GLMTemplate(['{{SYSTEM}}'], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'], [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation, GLMTemplate([], ['{{QUERY}}'], None, [['eos_token_id']]), is_generation=True)

register_template(
    TemplateType.chatglm3,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'], None, ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.chatglm4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'],
                None, ['<|system|>\n{{SYSTEM}}'],
                tools_prompt='glm4',
                tool_prompt=['<|observation|>\n{{QUERY}}<|assistant|>\n']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(
    TemplateType.codegeex4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|endoftext|>'], codegeex4_system,
                ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.deepseek,
    Template([['bos_token_id']], ['User: {{QUERY}}\n\nAssistant:'], [['eos_token_id']], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))
register_template(
    TemplateType.numina_math,
    Template([['bos_token_id']], ['### Problem: {{QUERY}}\n### Solution: '], ['\n'], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}']))
register_template(
    TemplateType.deepseek2,
    Template([[100000]], ['User: {{QUERY}}\n\nAssistant:'], [[100001]], [[100001]], None, [[100000], '{{SYSTEM}}\n\n']))
register_template(
    TemplateType.deepseek2_5,
    Template(['<｜begin▁of▁sentence｜>'], ['<｜User｜>{{QUERY}}<｜Assistant｜>'], ['<｜end_of_sentense｜>'],
             ['<｜end_of_sentense｜>'], None, ['<｜begin▁of▁sentence｜>{{SYSTEM}}']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(
    TemplateType.longwriter_llama3,
    Template(['[INST]'], ['{{QUERY}}[/INST]'], ['[INST]'], ['<|end_of_text|>'], None,
             ['<<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(TemplateType.mistral_nemo,
                  Template(['<s>[INST] '], ['{{SYSTEM}}\n\n', '{{QUERY}}[/INST]'], ['</s>[INST] '], ['</s>']))


class Llama3TemplateMixin:
    system = None

    def __init__(self):
        Template.__init__(
            self, ['<|begin_of_text|>'], [
                '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ], ['<|eot_id|>'], ['<|eot_id|>'],
            self.system, ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'],
            tools_prompt='toolbench',
            tool_prompt=[
                '<|start_header_id|>tool<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ])


class Llama3Template(Llama3TemplateMixin, Template):
    pass


class ReflectionTemplate(Llama3TemplateMixin, Template):
    system = ('You are a world-class AI system, capable of complex reasoning and reflection. '
              'Reason through the query inside <thinking> tags, and then provide your final '
              'response inside <output> tags. If you detect that you made a mistake in your reasoning '
              'at any point, correct yourself inside <reflection> tags.')


register_template(TemplateType.reflection, ReflectionTemplate())
register_template(TemplateType.llama3, Llama3Template())


class Llama3_2TemplateMixin:
    system = None

    def __init__(self):
        now = datetime.now()
        date_string = now.strftime('%d %b %Y')
        date_prompt = f'Cutting Knowledge Date: December 2023\nToday Date: {date_string}'
        Template.__init__(
            self, [
                f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{date_prompt}\n\n'
                '{{SYSTEM}}<|eot_id|>'
            ], [
                '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ], ['<|eot_id|>'], ['<|eot_id|>'],
            self.system,
            tools_prompt='toolbench',
            tool_prompt=[
                '<|start_header_id|>tool<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
                '<|start_header_id|>assistant<|end_header_id|>\n\n'
            ])


class Llama3_2Template(Llama3_2TemplateMixin, Template):
    pass


register_template(TemplateType.llama3_2, Llama3_2Template())


class Llama3_2VisionTemplateMixin:

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<|image|>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from transformers.models.mllama.processing_mllama import (get_cross_attention_token_mask,
                                                                  convert_sparse_cross_attention_mask_to_dense)
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        if images:
            input_ids = inputs['input_ids']
            processor = self.tokenizer.processor
            image_features = processor.image_processor(images, return_tensors='pt')
            num_tiles = image_features.pop('num_tiles')
            inputs.update(image_features)

            cross_attention_token_mask = [get_cross_attention_token_mask(input_ids, processor.image_token_id)]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=processor.image_processor.max_image_tiles,
                length=len(input_ids),
            )
            inputs['cross_attention_mask'] = torch.tensor(cross_attention_mask)

        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        for key in ['aspect_ratio_ids', 'aspect_ratio_mask']:
            value = [b[key] for b in batch if b.get(key) is not None]
            if value:
                res[key] = torch.concat(value)

        cross_attention_mask = [
            b['cross_attention_mask'][0] for b in batch if b.get('cross_attention_mask') is not None
        ]
        if cross_attention_mask:
            res['cross_attention_mask'] = self.pad_sequence(cross_attention_mask, 0, self.padding_side)
        return res


class Llama3_2VisionTemplate(Llama3_2VisionTemplateMixin, Llama3Template):
    pass


class Llama3_2VisionGenerationTemplate(Llama3_2VisionTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.llama3_2_vision, Llama3_2VisionTemplate(), lazy_tokenize=True)
register_template(TemplateType.llama3_2_vision_generation, Llama3_2VisionGenerationTemplate(), lazy_tokenize=True)


class Llama3_1OmniTemplate(Llama3Template):
    system = ('You are a helpful language and speech assistant. '
              'You are able to understand the speech content that the user provides, '
              'and assist the user with a variety of tasks using natural language.')

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'audio'
        return [[-200]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import whisper
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        audios = example['audios']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['_data'] = {'input_ids': torch.tensor(input_ids)[None]}
        if labels is not None:
            inputs['_data']['labels'] = torch.tensor(labels)[None]
        if audios:
            audios = load_batch(audios, whisper.load_audio)
            n_mels = get_env_args('n_mels', int, 128)
            for i, audio in enumerate(audios):
                audio = whisper.pad_or_trim(audio)
                audios[i] = whisper.log_mel_spectrogram(audio, n_mels=n_mels).permute(1, 0)
            audios = torch.stack(audios)
            inputs['_data'].update({'speech': audios, 'speech_lengths': torch.tensor([[audios.shape[1]]])})

        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        speech = data.get('speech')
        input_ids = data['input_ids']
        labels = data.get('labels')
        if speech is not None:
            speech_lengths = data['speech_lengths']
            speech = speech.to(model.dtype)
            inputs_embeds, labels = model.prepare_inputs_labels_for_speech_and_text(input_ids, None, None, None, labels,
                                                                                    speech, speech_lengths)[4:]
        else:
            inputs_embeds = model.get_model().embed_tokens(input_ids)
        res = {'inputs_embeds': inputs_embeds[0]}
        if labels is not None:
            res['labels'] = labels[0]
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(TemplateType.llama3_1_omni, Llama3_1OmniTemplate(), lazy_tokenize=True)

OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.')
register_template(
    TemplateType.openbuddy,
    Template([], ['User: {{QUERY}}\nAssistant:'], ['\n'], [['eos_token_id']],
             OPENBUDDY_DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))

OPENBUDDY2_DEFAULT_SYSTEM = (
    'You(assistant) are a helpful, respectful and honest INTP-T AI Assistant named Buddy. '
    'You are talking to a human(user).\nAlways answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any harmful, political, religious, unethical, racist, '
    'sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2023-04.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'not related to GPT or OpenAI')

register_template(
    TemplateType.openbuddy2,
    Template([], ['<|role|>user<|says|>{{QUERY}}<|end|>\n<|role|>assistant<|says|>'], ['<|end|>\n'], ['<|end|>'],
             OPENBUDDY2_DEFAULT_SYSTEM, ['<|role|>system<|says|>{{SYSTEM}}<|end|>\n'],
             auto_add_bos=True))

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'], INTERNLM_SYSTEM,
             ['<s><|System|>:{{SYSTEM}}\n']))

_T = TypeVar('_T')

_log_set = set()  # log once


def get_env_args(args_name: str,
                 type_func: Callable[[str], _T],
                 default_value: Optional[_T],
                 compat_args_names: Optional[List[str]] = None) -> Optional[_T]:
    # compat_args_names: compatibility
    if compat_args_names is None:
        compat_args_names = []
    args_name_list = [args_name] + compat_args_names
    for args_name in args_name_list:
        args_name_upper = args_name.upper()
        value = os.getenv(args_name_upper)
        if value is not None:
            value = type_func(value)
            log_info = f'Using environment variable `{args_name_upper}`, Setting {args_name}: {value}.'
            break
    else:
        args_name = args_name_list[0]
        args_name_upper = args_name.upper()
        value = default_value
        log_info = (f'Setting {args_name}: {default_value}. '
                    f'You can adjust this hyperparameter through the environment variable: `{args_name_upper}`.')
    if log_info not in _log_set:
        _log_set.add(log_info)
        logger.info(log_info)
    return value


class Internlm2Template(ChatmlTemplate):
    system = INTERNLM_SYSTEM


register_template(TemplateType.internlm2, Internlm2Template())


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


class InternLMXComposer2Template(Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
        'Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.')
    image_placeholder = ['</s>']

    def __init__(self, version):
        prefix = ['<s>']
        prompt = ['[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n']
        chat_sep = ['[UNUSED_TOKEN_145]\n']
        suffix = ['[UNUSED_TOKEN_145]']
        system_prefix = ['<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n']
        super().__init__(prefix, prompt, chat_sep, suffix, self.INTERNLM_XCOMPOSER_SYSTEM, system_prefix)
        self.version = version

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        dtype = self.model.dtype
        images = example.get('images') or []

        if self.version == 'v2.5':
            hd_num = 24
            if len(images) > 1:
                hd_num = 6
            hd_num = get_env_args('hd_num', int, hd_num)
            Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', self.tokenizer.model_dir)
            images = [Image_transform(image, hd_num=hd_num) for image in images]
        elif self.version == 'v2-4khd':
            hd_num = 55
            hd_num = get_env_args('hd_num', int, hd_num)
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', self.tokenizer.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        images = [self.model.vis_processor(image).to(dtype) for image in images]
        inputs['_data'] = {'input_ids': inputs['input_ids'], 'labels': inputs['labels'], 'images': images}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        input_ids = data['input_ids']
        labels = data['labels']
        images = data['images']
        if len(images) > 0:  # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            labels.append(2)
        else:
            labels = []
        res_inputs_embeds = []
        res_labels = []
        wrap_im_mask = []
        pre_i, i, idx = 0, 0, 0
        device = model.device
        internlm2_model = model.model
        if not hasattr(internlm2_model, 'tok_embeddings'):
            internlm2_model = internlm2_model.model
        tok_embeddings = internlm2_model.tok_embeddings
        if len(images) > 0:
            images = torch.concat([model.img2emb(image[None])[0] for image in images], dim=0)
        while i < len(input_ids):
            if input_ids[i] == 2:  # replace_token
                res_input_ids = torch.tensor([1] + input_ids[pre_i:i], device=device)
                res_inputs_embeds.append(tok_embeddings(res_input_ids[None])[0])
                wrap_im_mask += [0] * len(res_input_ids)
                res_labels += [-100] + labels[pre_i:i]
                if len(images) > 0 and idx < images.shape[0]:
                    res_inputs_embeds.append(images[idx].to(device))
                    wrap_im_mask += [1] * images.shape[1]
                    res_labels += [-100] * images.shape[1]
                idx += 1
                i += 1
                pre_i = i
                continue
            i += 1
        if len(labels) == 0:
            res_labels = None
        res_inputs_embeds = torch.concat(res_inputs_embeds, dim=0)
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool, device=device)[None]
        return {'inputs_embeds': res_inputs_embeds, 'im_mask': wrap_im_mask, 'labels': res_labels}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        if 'im_mask' in batch[0]:
            im_mask = [b['im_mask'][0] for b in batch]
            im_mask = self.pad_sequence(im_mask, 0, self.padding_side)
            res['im_mask'] = im_mask
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(
    TemplateType.internlm_xcomposer2, InternLMXComposer2Template(version='v2'), use_model=True, lazy_tokenize=True)


class InternLMXComposer2_5Template(InternLMXComposer2Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model '
        'that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively '
        'based on the provided image.')


register_template(
    TemplateType.internlm_xcomposer2_5,
    InternLMXComposer2_5Template(version='v2.5'),
    use_model=True,
    lazy_tokenize=True)

register_template(
    TemplateType.internlm_xcomposer2_4khd,
    InternLMXComposer2_5Template(version='v2-4khd'),
    use_model=True,
    lazy_tokenize=True)


class InternvlTemplate(Template):
    system = 'You are an AI assistant whose name is InternLM (书生·浦语).'
    num_image_token = 256

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'],
                         auto_add_bos=True)

    def replace_tag(self, media_type, index, example) -> List[Context]:
        if self._is_vllm:
            image_context = ['<img><image></img>\n']
        else:
            image_context = ['<img>', [-100], '</img>\n']
        return image_context

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        pixel_values = None
        images = example.get('images')
        if images:
            labels = inputs.get('labels')
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 12)
            pixel_values_images = [transform_image(image, input_size, max_num) for image in images]
            pixel_values = torch.cat(pixel_values_images, dim=0).to(self.model.dtype)
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * image_bs
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels
        inputs['_data'] = {'input_ids': torch.tensor(input_ids), 'pixel_values': pixel_values}
        inputs.pop('loss_scale', None)
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        input_ids = data['input_ids']
        inputs_embeds = embedding(input_ids[None])[0].to(device=device)
        pixel_values = data['pixel_values']
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            vit_embeds = model.extract_feature(pixel_values).to(device=device)
            selected = (input_ids == self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        elif is_deepspeed_enabled():
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=inputs_embeds.dtype)
            vit_embeds = model.extract_feature(dummy_pixel_values).to(device=device)
            inputs_embeds += vit_embeds.mean() * 0.
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


def _replace_video2image(load_video_func, example, replace_tag) -> List[Context]:
    context_list = []
    video_index = example['video_index']
    video = example['videos'][video_index]
    images = example['images']
    image_index = example['image_index']
    new_images = load_video_func(video)
    example['images'] = images[:image_index] + new_images + images[image_index:]
    for i in range(len(new_images)):
        context_list += replace_tag(i)
    example['image_index'] += len(new_images)
    return context_list


class Internvl2Template(InternvlTemplate):
    video_segments = 8
    system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'

    def replace_tag(self, media_type, index, example) -> List[Context]:
        image_context = super().replace_tag('image', index, example)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            video_segments = get_env_args('video_segments', int, self.video_segments)
            load_video = partial(load_video_internvl, num_segments=video_segments)
            return _replace_video2image(load_video, example, lambda i: [f'Frame{i + 1}: '] + image_context)

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            if isinstance(object_['bbox'][0], list):
                all_objects = '<box> ['
                for sub_object in object_['bbox']:
                    all_objects += (f'[{sub_object[0]}, {sub_object[1]}, ' f'{sub_object[2]}, {sub_object[3]}],')
                all_objects = all_objects[:-1]
                all_objects += '] </box>'
                return [all_objects]
            else:
                return [
                    f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                    f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
                ]
        else:
            return ['<bbox>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(InternvlTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        labels = inputs.get('labels')
        images = example.get('images')
        if images:
            has_video = bool(example.get('videos'))
            input_size = get_env_args('input_size', int, 448)
            max_num = get_env_args('max_num', int, 1 if has_video else 12)
            pixel_values = [transform_image(image, input_size, max_num) for image in images]
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values).to(self.model.dtype)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = self.tokenizer.encode(
                '<IMG_CONTEXT>', add_special_tokens=False) * self.num_image_token * num_patch
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['_data'] = {'input_ids': torch.tensor(input_ids), 'pixel_values': pixel_values}
        inputs.pop('loss_scale', None)
        return inputs, {}


class InternvlPhi3TemplateMixin:

    def __init__(self):
        Template.__init__(
            self, [], ['<|user|>\n{{QUERY}}<|end|><|assistant|>\n'], ['<|end|>'], ['<|end|>'],
            getattr(self, 'system', None), ['<|system|>\n{{SYSTEM}}<|end|>'],
            auto_add_bos=True)
        self.padding_side = 'left'


class InternvlPhi3Template(InternvlPhi3TemplateMixin, InternvlTemplate):
    system = 'You are an AI assistant whose name is Phi-3.'


class Internvl2Phi3Template(InternvlPhi3TemplateMixin, Internvl2Template):
    pass


register_template(
    TemplateType.internvl, InternvlTemplate(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.internvl_phi3, InternvlPhi3Template(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(TemplateType.internvl2, Internvl2Template(), use_model=True, lazy_tokenize=True)

register_template(TemplateType.internvl2_phi3, Internvl2Phi3Template(), use_model=True, lazy_tokenize=True)


class FlorenceTemplate(Template):
    compute_per_round_loss = False
    output_prompt_answer = True

    def __init__(self):
        super().__init__(['<s>'], ['{{QUERY}}</s>'], None, ['</s>'])
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
        }
        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) == 1, 'Florence series models only supports input with a single image.'

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        return

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        object_ = example['objects'][index]
        if isinstance(object_['bbox'][0], list):
            all_objects = ''
            for sub_object in object_['bbox']:
                x1, y1, x2, y2 = sub_object
                all_objects += f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>,'
            return [all_objects[:-1]]
        else:
            x1, y1, x2, y2 = object_['bbox']
            return [f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = example['query']
        processor = self.tokenizer.processor
        example['query'] = processor._construct_prompts([query])[0]
        inputs, _ = super()._encode(example)
        input_ids = inputs['prompt_input_ids']
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        labels = inputs['answer_labels']
        if labels is not None:
            labels = [0] + labels
        pixel_values = processor.image_processor(images, return_tensors='pt')['pixel_values'].to(self.model.dtype)
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'pixel_values': pixel_values,
            }
        }
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.get_input_embeddings()(data['input_ids'])
        image_features = model._encode_image(data['pixel_values'])
        inputs_embeds, _ = model._merge_input_ids_with_image_features(image_features, inputs_embeds)
        return {'inputs_embeds': inputs_embeds[0]}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids

    def post_process_generate_response(self, response, example):
        if isinstance(example['images'], list):
            example['images'] = example['images'][0]
        image = load_image(example['images'])
        return json.dumps(
            self.tokenizer.processor.post_process_generation(
                response, task=example['query'], image_size=(image.width, image.height)))


register_template(
    TemplateType.florence,
    FlorenceTemplate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    stream=False)

register_template(TemplateType.xverse,
                  Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '], [['eos_token_id']], [['eos_token_id']]))
register_template(TemplateType.yuan, Template([], ['{{QUERY}}<sep>'], None, [['eos_token_id']]))
register_template(TemplateType.ziya,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['<human>:{{QUERY}}\n<bot>:'], ['\n'], [['eos_token_id']]))

register_template(TemplateType.skywork,
                  Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None, ['[SEP]</s>']))

register_template(TemplateType.bluelm,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'], [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse_codellama,
    Template(['{{SYSTEM}}'], ['<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'], [],
             [['eos_token_id']]))

register_template(
    TemplateType.codefuse,
    Template([], ['<s>human\n{{QUERY}}\n<s>bot\n'], [['eos_token_id'], '\n'], [['eos_token_id']], None,
             ['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateType.deepseek_coder,
    Template(['{{SYSTEM}}'], ['### Instruction:\n{{QUERY}}\n### Response:\n'], ['\n<|EOT|>\n'], ['\n<|EOT|>'],
             ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
              'developed by Deepseek Company, and you only answer questions related to computer science. '
              'For politically sensitive questions, security and privacy issues, '
              'and other non-computer science questions, you will refuse to answer\n')))


class LlavaHfTemplate(Template):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if version.parse(transformers.__version__) < version.parse('4.43.0'):
            self.padding_side = 'left'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        if images:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


class Llava1_6Llama3Template(LlavaHfTemplate):
    default_system = 'You are a helpful language and vision assistant. ' \
                     'You are able to understand the visual content that the user provides, ' \
                     'and assist the user with a variety of tasks using natural language.'

    def __init__(self):
        super().__init__(['<|begin_of_text|>'], [
            '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ], ['<|eot_id|>'], ['<|eot_id|>'], None,
                         ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs['pixel_values'].shape) == 5:  # (1, num_patch, 3, H/W, W/H)
            inputs['pixel_values'] = torch.squeeze(inputs['pixel_values'], dim=0)  # (num_patch, 3, H/W, W/H)
        return inputs, {}


register_template(TemplateType.llava_next_llama3, Llava1_6Llama3Template(), use_model=True, lazy_tokenize=True)


class LlavaVideoTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:

        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = example['videos'][index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            example['videos'][index] = load_video_llava(example['videos'][index])
            return ['<video>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        videos_path = example.get('videos') or []
        if len(videos_path) > 0:
            video_processor = self.tokenizer.processor.video_processor
            video_inputs = video_processor(videos_path, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(
    TemplateType.llava_next_video,
    LlavaVideoTemplate(['<s>{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['</s>']),
    use_model=True,
    lazy_tokenize=True)

register_template(
    TemplateType.llava_next_video_yi,
    LlavaVideoTemplate(['{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['<|im_end|>']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)


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


class Idefics3Template(Template):

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        processor = self.tokenizer.processor
        prompt = self.tokenizer.decode(inputs['input_ids'])
        if images:
            image_inputs = processor(text=prompt, images=images, return_tensors='pt', add_special_tokens=False)
            image_token = 128257  # <image>
            inputs['input_ids'], inputs['labels'] = align_image_inputs(inputs['input_ids'], inputs['labels'],
                                                                       image_inputs['input_ids'][0], image_token)
            inputs['pixel_values'] = image_inputs['pixel_values']
        return inputs, {}


register_template(
    TemplateType.idefics3,
    Idefics3Template(['<|begin_of_text|>'], ['User:{{QUERY}}<end_of_utterance>\nAssistant:'], ['<end_of_utterance>\n'],
                     ['<end_of_utterance>'], None, ['System:{{SYSTEM}}<end_of_utterance>\n']),
    use_model=True,
    lazy_tokenize=True)


class Llava1_5Template(LlavaHfTemplate):

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}}\nASSISTANT:'], ['</s>'], ['</s>'])


register_template(TemplateType.llava1_5, Llava1_5Template(), use_model=True, lazy_tokenize=True)


class LLavaTemplate(Template):

    def __init__(self):
        # This template follows: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L350
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'],
                         None, ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, self.model.config)
            inputs['images'] = images_tensor.to(model.dtype).squeeze(0)
            inputs['image_sizes'] = image_sizes
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'Llava does not support mix-batch nlp dataset and multi-modal dataset'
        return res

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


class Llava1_6Template(LlavaHfTemplate):

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)  # 5d -> 4d
        res = super().data_collator(batch, padding_to)
        return res


class Llava1_6MistralTemplate(Llava1_6Template):

    def __init__(self):
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s>'], ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


class Llava1_6VicunaTemplate(Llava1_6Template):
    system = ('A chat between a curious human and an artificial intelligence assistant. '
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'],
                         self.system,
                         system_prefix=['<s>{{SYSTEM}} '])


register_template(TemplateType.llava_mistral, Llava1_6MistralTemplate(), use_model=True, lazy_tokenize=True)

register_template(TemplateType.llava_vicuna, Llava1_6VicunaTemplate(), use_model=True, lazy_tokenize=True)


class LLava1_6YiTemplate(Llava1_6Template):

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        if self._is_vllm:
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, example)


register_template(TemplateType.llava_yi, LLava1_6YiTemplate(), use_model=True, lazy_tokenize=True)


class Llama3LlavaNextHfTemplate(Llama3TemplateMixin, Llava1_6Template):
    pass


register_template(TemplateType.llama3_llava_next_hf, Llama3LlavaNextHfTemplate(), use_model=True, lazy_tokenize=True)


class LlavaQwenHfTemplate(QwenTemplateMixin, Llava1_6Template):
    pass


register_template(TemplateType.llava_qwen_hf, LlavaQwenHfTemplate(), use_model=True, lazy_tokenize=True)


class LlavaOneVisonTemplate(QwenTemplateMixin, Llava1_6Template):
    system = None

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 151646)  # <image>
        processor = self.tokenizer.processor
        if images:
            image_processor = processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            height, width = image_inputs['pixel_values'][0].shape[-2:]
            added_tokens_len = 0
            for idx, pixel_v, image_size in zip(idx_list, image_inputs['pixel_values'], image_inputs['image_sizes']):
                orig_height, orig_width = image_size
                num_image_tokens = processor._get_number_of_features(orig_height, orig_width, height, width)
                input_ids = input_ids[:added_tokens_len
                                      + idx] + [151646] * num_image_tokens + input_ids[added_tokens_len + idx + 1:]
                if labels is not None:
                    labels = labels[:added_tokens_len + idx] + [-100] * num_image_tokens + labels[added_tokens_len + idx
                                                                                                  + 1:]
                added_tokens_len += num_image_tokens - 1
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(TemplateType.llava_onevision_qwen, LlavaOneVisonTemplate(), use_model=True, lazy_tokenize=True)


class LLavaLlamaTemplate(Llama3Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example):
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images')
        if raw_image:
            pixel_values = self.tokenizer.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            inputs['pixel_values'] = pixel_values.to(self.model.dtype)
        return inputs, {}


register_template(TemplateType.llava_llama_instruct, LLavaLlamaTemplate(), use_model=True, lazy_tokenize=True)


class MolmoTemplate(Template):
    system = None
    image_placeholder = ['<|image|>']
    DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
    DEFAULT_IM_START_TOKEN = '<im_start>'
    DEFAULT_IM_END_TOKEN = '<im_end>'
    DEFAULT_IM_COL_TOKEN = '<im_col>'

    def __init__(self):
        Template.__init__(self, [], [' User: {{QUERY}} Assistant:'], ['<|endoftext|>'], ['<|endoftext|>'], self.system)
        self.processor_kwargs = {
            'images_kwargs': {
                'max_crops': 12,
                'overlap_margins': [4, 4],
                'base_image_input_size': [336, 336],
                'image_token_length_w': 12,
                'image_token_length_h': 12,
                'image_patch_size': 14,
                'image_padding_mask': True,
            },
            'text_kwargs': {
                'style': 'long_caption',
                'system_prompt': 'none',
                'message_format': 'role',
                'always_start_with_space': True,
                'sequence_length': 1536,
                'padding': False,
            }
        }

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        # image
        raw_image = example.get('images', None)
        res = {}
        labels = inputs['labels']
        if raw_image:
            image_id = self.tokenizer.convert_tokens_to_ids(self.image_placeholder)
            idx_list = _findall(inputs['input_ids'], image_id)
            res = self._process_images(raw_image, inputs['input_ids'], idx_list, labels)
            import numpy as np
            if 'image_input_idx' in res:
                # Shift patch mapping up by one since we added BOS
                image_input_idx = res['image_input_idx']
                res['image_input_idx'] = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)
            inputs['input_ids'] = res.pop('input_ids').tolist()
            if labels:
                inputs['labels'] = [-100] + res.pop('labels')  # add one label for BOS

            for k, v in res.items():
                res[k] = torch.from_numpy(v).unsqueeze(0)
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        inputs['input_ids'] = [bos] + inputs['input_ids']
        res.update({'input_ids': inputs['input_ids']})
        # prepare meta inputs
        inputs.update(self.prepare_meta_inputs(res))

        return inputs, {}

    def _process_images(self, images: List, tokens: List, idx_list: List = None, labels: List = None) -> torch.Tensor:
        from PIL import ImageOps
        from PIL.Image import Image
        import numpy as np
        if images is not None:
            image_arrays = []
            for image in images:
                if isinstance(image, Image):
                    image = image.convert('RGB')
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images = image_arrays
            # For now only support inserting images at the start
        if idx_list is None:
            idx_list = [-1] * len(images)
        image_patch_token_id = self.tokenizer.processor.special_token_ids[self.DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.tokenizer.processor.special_token_ids[self.DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.tokenizer.processor.special_token_ids[self.DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.tokenizer.processor.special_token_ids[self.DEFAULT_IM_END_TOKEN]
        sequence_length = self.processor_kwargs['text_kwargs']['sequence_length']
        res = self.tokenizer.processor.image_processor.multimodal_preprocess(
            images=images,
            image_idx=idx_list,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=sequence_length,
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
            **self.processor_kwargs['images_kwargs'])
        if labels is not None:
            new_labels = []
            cur_idx = 0
            for input_id in res['input_ids']:
                if input_id in (image_start_token_id, image_end_token_id, image_col_token_id, image_patch_token_id):
                    new_labels.append(-100)
                    if tokens[cur_idx] == self.tokenizer.convert_tokens_to_ids(self.image_placeholder)[0]:
                        cur_idx += 1
                else:
                    new_labels.append(labels[cur_idx])
                    cur_idx += 1
            res['labels'] = new_labels
        return res

    def prepare_meta_inputs(self, data: Any) -> Dict[str, Any]:

        # prepare batch inputs
        input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
        generation_config = self.model.generation_config

        batch_size, seq_len = input_ids.shape
        attention_mask = None
        max_new_tokens = generation_config.max_new_tokens
        assert max_new_tokens is not None
        mask_len = seq_len + max_new_tokens if self.model.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.model.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1, min=0)
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if self._is_training:
            # no batch_size before data_collator
            attention_mask = attention_mask.squeeze(0)
            position_ids = position_ids.squeeze(0)
        data.update({
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'append_last_valid_logits': append_last_valid_logits,
        })
        if 'images' in data:
            data['images'] = data['images'].to(self.model.dtype)
        return data

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        # prepare batch attention_mask
        attention_mask = res['attention_mask']
        generation_config = self.model.generation_config
        max_new_tokens = generation_config.max_new_tokens
        batch_size, seq_len = attention_mask.shape
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((batch_size, max_new_tokens))],
            dim=1,
        )
        # prepare batchfy inputs
        keys = ['images', 'image_input_idx', 'image_masks', 'append_last_valid_logits']
        for key in keys:
            batch_input = [b[key] for b in batch if b.get(key) is not None]
            res[key] = torch.concat(batch_input)

        return res


register_template(TemplateType.molmo, MolmoTemplate(), lazy_tokenize=True, use_model=True)


class Emu3ChatTemplate(Template):
    system = 'You are a helpful assistant.'
    image_placeholder = ['<|image token|>']

    def __init__(self):
        Template.__init__(self, [['bos_token_id'], '{{SYSTEM}}'], [' User: {{QUERY}}. Assistant:'], [['eos_token_id']],
                          [['eos_token_id']], self.system)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        # image
        raw_image = example.get('images', None)
        if raw_image:
            inputs['_data'] = {'raw_image': raw_image, 'input_ids': inputs['input_ids'], 'labels': inputs['labels']}

        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        raw_images = data['raw_image']
        input_ids = data['input_ids']
        labels = data['labels']
        image_tokens = self.tokenizer.processor.tokenize_image(raw_images)
        image_prompts = []
        idxs = _findall(input_ids, self.tokenizer.encode(self.image_placeholder))
        # Create image prompts
        for i in range(len(raw_images)):
            h, w = image_tokens[i].shape
            imgstr = self.tokenizer.processor.to_imgstr(image_tokens[i])
            image_prompt = (
                self.tokenizer.boi_token + self.tokenizer.processor.prefix_template.format(H=h, W=w)
                + self.tokenizer.img_token + imgstr + self.tokenizer.eol_token + self.tokenizer.eof_token
                + self.tokenizer.eoi_token)
            image_prompts.append(self.tokenizer.encode(image_prompt))
        added_tokens_len = 0
        # Insert image tokens into input_ids
        for idx, img_tokens in zip(idxs, image_prompts):
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1

        return {'input_ids': input_ids, 'labels': labels}


register_template(TemplateType.emu3_chat, Emu3ChatTemplate(), lazy_tokenize=True, use_model=True)


class PaliGemmaTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}\n'], None, ['<eos>'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        if self._is_vllm:
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.tokenizer.processor.image_seq_length + '<bos>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images')
        processor = self.tokenizer.processor
        if inputs['labels'] is not None:
            n = upper_bound(0, len(inputs['labels']), lambda idx: inputs['labels'][idx] == -100)
            n2 = len(inputs['labels']) - n
            inputs['token_type_ids'] = [0] * n + [1] * n2
        else:
            inputs['token_type_ids'] = [0] * len(inputs['input_ids'])
        if raw_image:
            model_inputs = processor(text=example['query'], images=raw_image[0], return_tensors='pt')
            inputs['pixel_values'] = model_inputs['pixel_values']
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.paligemma, PaliGemmaTemplate(), infer_media_type='dialogue', lazy_tokenize=True, is_generation=True)


class Phi3Template(Template):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'], ['<|end|>\n'], ['<|end|>'],
                         None, ['<|system|>\n{{SYSTEM}}<|end|>\n'],
                         auto_add_bos=True)


register_template(TemplateType.phi3, Phi3Template())


class Phi3VisionTemplate(Phi3Template):
    image_placeholder = ['<|image|><s>\n']  # <|image|>\n

    def replace_tag(self, media_type, index, example) -> List[Context]:
        if self._is_vllm:
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, example)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = example.get('images') or []
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.tokenizer.processor
            inputs.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = inputs.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -i - 1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * num_img_tokens[i]
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}


register_template(TemplateType.phi3_vl, Phi3VisionTemplate(), lazy_tokenize=True)


class Llama3LlavaNextTemplate(Llama3TemplateMixin, LLavaTemplate):
    system = 'You are a helpful language and vision assistant. ' \
             'You are able to understand the visual content that the user provides, ' \
             'and assist the user with a variety of tasks using natural language.'


register_template(TemplateType.llama3_llava_next, Llama3LlavaNextTemplate(), use_model=True, lazy_tokenize=True)


class LLavaQwenTemplate(QwenTemplateMixin, LLavaTemplate):
    pass


register_template(TemplateType.llava_qwen, LLavaQwenTemplate(), use_model=True, lazy_tokenize=True)


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


class DeepseekVLTemplate(Template):
    DEEPSEEK_VL_SYSTEM = ('You are a helpful language and vision assistant. '
                          'You are able to understand the visual content that the user provides, '
                          'and assist the user with a variety of tasks using natural language.')

    image_placeholder = ['<image_placeholder>']

    def __init__(self):
        super().__init__(['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'], ['User: {{QUERY}}\n\nAssistant:'],
                         ['<｜end▁of▁sentence｜>'], ['<｜end▁of▁sentence｜>'], self.DEEPSEEK_VL_SYSTEM)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        is_janus = getattr(self, 'is_janus', False)

        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        processor = self.tokenizer.processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = _findall(input_ids, processor.image_id)  # '<image_placeholder>'
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            image_tokens = [processor.image_id] * processor.num_image_tokens
            if is_janus:
                image_tokens = [processor.image_start_id] + image_tokens + [processor.image_end_id]
            new_input_ids += image_tokens
            new_labels += [-100] * len(image_tokens)
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        if is_janus:
            from janus.models.processing_vlm import VLChatProcessorOutput
        else:
            from deepseek_vl.models.processing_vlm import VLChatProcessorOutput

        images_outputs = processor.image_processor(images, return_tensors='pt')
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(new_input_ids),
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=torch.tensor([processor.num_image_tokens] * len(idx_list)))
        batched_output = dict(processor.batchify([output]))
        batched_output['pixel_values'] = batched_output['pixel_values'].to(dtype=self.model.dtype)
        inputs = {'input_ids': new_input_ids, 'labels': new_labels, '_data': batched_output}
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds = model.prepare_inputs_embeds(**data)[0]
        return {'inputs_embeds': inputs_embeds}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


register_template(TemplateType.deepseek_vl, DeepseekVLTemplate(), use_model=True, lazy_tokenize=True)


class DeepseekJanus(DeepseekVLTemplate):
    is_janus = True
    image_placeholder = ['<image_placeholder>\n']


register_template(TemplateType.deepseek_janus, DeepseekJanus(), use_model=True, lazy_tokenize=True)

register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'], ['</s>'], None,
             ['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateType.sus,
    Template(['{{SYSTEM}}'], ['### Human: {{QUERY}}\n\n### Assistant: '], ['<|endoftext|>'], ['<|endoftext|>']))

register_template(TemplateType.orion,
                  Template(['<s>{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: </s>'], ['</s>'], ['</s>']))


class CogTemplate(Template):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return []

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        image = example.get('images') or []
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer, query=example['query'], history=example.get('history'), images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=dtype)] for cross_img in inputs2['cross_images']]
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.cogagent_chat,
    CogTemplate(['<s>'], [' [INST] {{QUERY}} [/INST] '], [], ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm,
    CogTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = example.get('videos') or []
        video = load_batch(videos_path, load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer,
            query=example['query'],
            history=example.get('history'),
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return inputs, {}


register_template(
    TemplateType.cogvlm2_video,
    Cog2VideoTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True,
    media_type='video')

register_template(TemplateType.minicpm, Template(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']))


def _remove_idx(arr: List[int], idx_list: List[int]) -> List[int]:
    res = []
    idx_set = set(idx_list)
    for i, x in enumerate(arr):
        if i not in idx_set:
            res.append(x)
    return res


class MiniCPMVTemplate(Template):
    is_v2_5 = False

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        if self._is_vllm:
            return ['(<image>./</image>)\n']
        else:
            return [[-100]]

    def check_example(self, example):
        images = example.get('images') or []
        if not self._is_vllm and not self._is_lmdeploy:
            assert len(images) == 1

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        idx_list.insert(0, -1)
        new_input_ids = []
        features = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            context_list = ['<image>', [-100], '</image>']
            feat = [x.squeeze() for x in images[i]['embeddings'].split(1)]
            grid = images[i].get('grid')
            if len(feat) > 1 and grid is not None:
                context_list.append('<slice>')
                for j in range(grid[1]):
                    if j > 0:
                        context_list.append('\n')
                    for _ in range(grid[0]):
                        context_list += ['<image>', [-100], '</image>']
                context_list.append('</slice>\n')
            new_input_ids += self._encode_context_list(context_list)[0]
            features += feat
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['images'] = features
        await super().prepare_lmdeploy_inputs(inputs)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        idx = idx_list[0]
        config = self.model.config
        tgt_sizes = None
        slice_mode = getattr(config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                image_processor = self.tokenizer.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = image_inputs['pixel_values']
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = self.model.get_slice_image_placeholder(images[0], self.tokenizer)
                pixel_values = [[self.model.transform(img) for img in images]]
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            input_tensor_ids = torch.tensor(input_ids)
            image_start_idx = torch.where(input_tensor_ids == self.tokenizer.im_start_id)[0]
            image_start_idx += 1
            image_end_idx = torch.where(input_tensor_ids == self.tokenizer.im_end_id)[0]
            valid_image_nums = max(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack(
                    [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)])
            ]
        else:
            placeholder = '<image>' + '<unk>' * config.query_num + '</image>\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + config.query_num]])]
            pixel_values = [[self.model.transform(images[0])]]
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'image_bound': image_bound,
                'pixel_values': pixel_values,
                'tgt_sizes': tgt_sizes
            }
        }
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        inputs_embeds, _ = model.get_vllm_embedding(data)
        return {'inputs_embeds': inputs_embeds[0]}

    @staticmethod
    def _get_generate_ids(generate_ids: List[int], input_token_len: int) -> List[int]:
        return generate_ids


class MiniCPMV2_6Template(QwenTemplateMixin, MiniCPMVTemplate):

    def check_example(self, example):
        pass

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 64)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        image_context = super().replace_tag('image', index, example)
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            return _replace_video2image(load_video, example, lambda i: image_context)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = Template._encode(self, example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        use_video = bool(example.get('videos'))
        is_plain_text = not images and not use_video
        use_image_id = True
        max_slice_nums = None

        if use_video:
            use_image_id = False
            max_slice_nums = 1  # or 2

        max_slice_nums = get_env_args('max_slice_nums', int, max_slice_nums)
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        idx_list.insert(0, -1)

        image_processor = self.tokenizer.processor.image_processor
        image_inputs = image_processor([images], return_tensors='pt',
                                       max_slice_nums=max_slice_nums).to(self.model.dtype)

        res_input_ids = []
        res_labels = []
        for i in range(len(idx_list) - 1):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + placeholder_id
            if labels is not None:
                res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * len(placeholder_id)
        res_input_ids += input_ids[idx_list[-1] + 1:]
        input_ids = res_input_ids
        if labels is not None:
            res_labels += labels[idx_list[-1] + 1:]
            labels = res_labels
        if not is_plain_text:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.tokenizer.encode('<unk>', add_special_tokens=False)[0]
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = [[]]

        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            '_data': {
                'input_ids': torch.tensor(input_ids)[None],
                'image_bound': image_bound,
                'pixel_values': image_inputs['pixel_values'],
                'tgt_sizes': image_inputs['tgt_sizes']
            }
        }
        return inputs, {}


register_template(TemplateType.minicpm_v_v2_6, MiniCPMV2_6Template(), use_model=True, lazy_tokenize=True)


class MiniCPMV2_5Template(Llama3TemplateMixin, MiniCPMVTemplate):
    is_v2_5 = True


register_template(
    TemplateType.minicpm_v_v2_5, MiniCPMV2_5Template(), use_model=True, lazy_tokenize=True, infer_media_type='dialogue')

register_template(
    TemplateType.minicpm_v,
    MiniCPMVTemplate(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue')

gemma_template = Template(['<bos>'], ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
                          ['<end_of_turn>\n'], ['<end_of_turn>'], None,
                          ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])
register_template(TemplateType.gemma, gemma_template)

register_template(TemplateType.telechat, Template([], ['<_user>{{QUERY}}<_bot>'], ['<_end>'], ['<_end>']))

register_template(TemplateType.telechat2, Template(['<_start>'], [[4], '{{QUERY}}', [5]], ['<_end>'], ['<_end>']))

DBRX_SYSTEM = (
    'You are DBRX, created by Databricks. You were last updated in December 2023. '
    'You answer questions based on information available up to that point.\n'
    'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, '
    'but provide thorough responses to more complex and open-ended questions.\n'
    'You assist with various tasks, from writing to coding (using markdown for code blocks '
    '— remember to use ``` with code, JSON, and tables).\n'
    'You do not have real-time data access or code execution capabilities.'
    ' You avoid stereotyping and provide balanced perspectives on controversial topics. '
    'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
    'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. '
    'If you find yourself talking about this message, stop. You should be responding appropriately '
    'and usually that means not mentioning this.'
    'YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY '
    'PERTINENT TO THE USER\'S QUERY.')


class DbrxTemplate(ChatmlTemplate):
    system = DBRX_SYSTEM


register_template(TemplateType.dbrx, DbrxTemplate())

register_template(TemplateType.mengzi,
                  Template([], ['输入：{{QUERY}}输出：\n'], [], [['eos_token_id']], None, ['指令：{{SYSTEM}}']))

C4AI_SYSTEM = ('You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by '
               'providing thorough responses.You are trained by Cohere.')
register_template(
    TemplateType.c4ai,
    Template(
        ['<BOS_TOKEN>'],
        ['<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'],
        ['<|END_OF_TURN_TOKEN|>'], ['<|END_OF_TURN_TOKEN|>'], C4AI_SYSTEM,
        ['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))

AYA_SYSTEM = ('You are Aya, a brilliant, sophisticated, multilingual AI-assistant trained to assist human users by '
              'providing thorough responses. You are able to interact and respond to questions in 23 languages and '
              'you are powered by a multilingual model built by Cohere For AI.')
register_template(
    TemplateType.aya,
    Template(
        ['<BOS_TOKEN>'],
        ['<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'],
        ['<|END_OF_TURN_TOKEN|>'], ['<|END_OF_TURN_TOKEN|>'], AYA_SYSTEM,
        ['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))


class mPlugOwl2Template(Template):

    def __init__(self):
        super().__init__(['{{SYSTEM}}'], ['USER: {{QUERY}}ASSISTANT:'], ['</s>'], [['eos_token_id']])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from mplug_owl2.mm_utils import process_images
        processor = self.tokenizer.processor
        images = example.get('images') or []
        for i, image in enumerate(images):
            # ref: https://modelscope.cn/models/iic/mPLUG-Owl2.1
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            images[i] = image
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if images:
            images = process_images(images, processor)
            images = images.to(self.model.dtype)
            return {'input_ids': input_ids, 'labels': labels, 'images': images}, {}
        else:
            return {'input_ids': input_ids, 'labels': labels}, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(
    TemplateType.mplug_owl2, mPlugOwl2Template(), infer_media_type='round', use_model=True, lazy_tokenize=True)


class mPlugOwl3Template(QwenTemplateMixin, Template):
    system = None

    def _get_image_token_list(self, cut_shape):
        processor = self.tokenizer.processor
        text = processor.image_processor.cut_prompt_template(img_token='<|image|>', h=cut_shape[0], w=cut_shape[1])
        text_list = text.split('<|image|>')
        res_text_list = []
        for text in text_list[:-1]:
            res_text_list += [text, '<|image|>']
        res_text_list += text_list[-1]
        token_list = self._encode_context_list(res_text_list)[0]
        return token_list

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        max_num_frames = get_env_args('max_num_frames', int, 16)
        load_video = partial(load_video_minicpmv_mplug_owl3, max_num_frames=max_num_frames)
        if media_type == 'image':
            return [[-100], '\n']
        elif media_type == 'video':
            return _replace_video2image(load_video, example, lambda i: [[-100]]) + ['\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        videos = example['videos']
        cut_enable = not videos
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        processor = self.tokenizer.processor
        inputs = {'_data': {}}
        if images:
            image_inputs = processor.image_processor(images, cut_enable=cut_enable, return_tensors='pt')
            added_tokens_len = 0
            cut_shapes = image_inputs['cut_shape'] or [None] * len(idx_list)
            image_token_list = self.tokenizer.encode('<|image|>', add_special_tokens=False)
            for idx, cut_shape in zip(idx_list, cut_shapes):
                if cut_shape:
                    token_list = self._get_image_token_list(cut_shape)
                else:
                    token_list = image_token_list
                input_ids = input_ids[:idx + added_tokens_len] + token_list + input_ids[added_tokens_len + idx + 1:]
                if labels:
                    labels = labels[:idx + added_tokens_len] + [-100] * len(token_list) + labels[added_tokens_len + idx
                                                                                                 + 1:]
                added_tokens_len += len(token_list) - 1
            image_token_idx = torch.tensor(_findall(input_ids, image_token_list))[None]
            _range = torch.arange(len(input_ids))[:, None]
            matrix = (_range > image_token_idx).sum(dim=1)
            media_offset = torch.stack([torch.zeros(matrix.shape[0], dtype=torch.long), matrix], dim=-1)[None]
            inputs['_data'].update({
                'pixel_values': image_inputs['pixel_values'],
                'media_offset': media_offset,
            })
        inputs['_data']['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def _post_encode(self, model, data: Any) -> Dict[str, Any]:
        if 'pixel_values' in data:
            pixel_values = data.pop('pixel_values')
            data['image_embeds'] = model.forward_image(pixel_values)
        return data

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        image_embeds = [b['image_embeds'] for b in batch if 'image_embeds' in b]
        if image_embeds:
            res['image_embeds'] = torch.concat(image_embeds)
        media_offset = []
        cusum_offset = 0

        for bi, b in enumerate(batch):
            if 'media_offset' in b:
                max_sequence_length = res['input_ids'].shape[1]
                curr_media_offset = b['media_offset']
                if curr_media_offset.shape[1] < max_sequence_length:
                    padding = curr_media_offset[:, -1:, :].expand(curr_media_offset.shape[0],
                                                                  max_sequence_length - curr_media_offset.shape[1],
                                                                  curr_media_offset.shape[2])
                    curr_media_offset = torch.concat([curr_media_offset, padding], dim=1)
                media_offset.append(curr_media_offset + cusum_offset)
                cusum_offset += image_embeds[bi].shape[0]

        # media_offset = [b['media_offset'] for b in batch if 'media_offset' in b]

        if media_offset:
            res['media_offset'] = torch.concat(media_offset)
        return res


register_template(TemplateType.mplug_owl3, mPlugOwl3Template(), use_model=True, lazy_tokenize=True)

register_template(TemplateType.wizardlm2_awq,
                  Template(['{{SYSTEM}}'], ['User:\n{{QUERY}}\n\nAssistant:\n'], ['\n\n'], ['</s>']))

_wizardlm2_system = ('A chat between a curious user and an artificial intelligence assistant. '
                     'The assistant gives helpful, detailed, and polite answers to the user\'s questions. ')
register_template(TemplateType.wizardlm2,
                  Template(['{{SYSTEM}}'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'], _wizardlm2_system))

register_template(TemplateType.atom,
                  Template(['{{SYSTEM}}'], ['<s>Human: {{QUERY}}\n</s><s>Assistant: '], ['</s>'], ['</s>']))


class RLHFTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        template_encode = self._old_encode
        inputs = {}
        tokenizer_kwargs = {}
        chosen_example, rejected_example = example, example.copy()
        rejected_example['response'] = example['rejected_response']
        if streaming:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example), {}
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example), {}
        else:
            chosen_inputs, chosen_tokenizer_kwargs = template_encode(chosen_example)
            rejected_inputs, rejected_tokenizer_kwargs = template_encode(rejected_example)

        if len(chosen_inputs) == 0 or len(rejected_inputs) == 0:
            return {}, {}
        for suffix, res in zip(['inputs', 'tokenizer_kwargs'], [inputs, tokenizer_kwargs]):
            for prefix in ['chosen', 'rejected']:
                data = locals()[f'{prefix}_{suffix}']
                for k, v in data.items():
                    res[f'{prefix}_{k}'] = v
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        _data_collator = self._old_data_collator
        new_batch = []
        for prefix in ['chosen_', 'rejected_']:
            for inputs in batch:
                new_inputs = {}
                for k, v in inputs.items():
                    if k.startswith(prefix):
                        new_k = k[len(prefix):]
                        new_inputs[new_k] = inputs[k]
                if len(new_inputs) > 0:
                    new_batch.append(new_inputs)
        assert len(new_batch) in {0, len(batch) * 2}, f'new_batch: {new_batch}'
        return _data_collator(new_batch or batch, padding_to)


class KTOTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = self._old_encode(example, streaming)
        if len(inputs) > 0:
            inputs['label'] = example['label']
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = {}
        for prefix in ['', 'KL_']:
            new_batch = []
            for b in batch:
                new_batch.append({'input_ids': b[f'{prefix}input_ids'], 'labels': b[f'{prefix}labels']})
            for k, v in self._old_data_collator(new_batch, padding_to).items():
                res[f'{prefix}completion_{k}'] = v
        res['label'] = [b['label'] for b in batch]
        return res


class PPOTemplateMixin:

    def encode(self: Template,
               example: Dict[str, Any],
               streaming: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = self._old_encode(example, streaming)
        if len(inputs) > 0:
            inputs.pop('labels')
        return inputs, tokenizer_kwargs

    def data_collator(self: Template, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        return self._old_data_collator(batch, padding_to)


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
    model=None,
    **kwargs,
) -> Template:
    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template._init_template(tokenizer, default_system, max_length, truncation_strategy, model=model, **kwargs)
    return template
