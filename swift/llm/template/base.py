# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import inspect
import os
import re
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial, wraps
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import get_logger
from peft import PeftModel
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.utils import dataclass_to_dict
from .agent import loss_scale_map, split_str_parts_by
from .template_inputs import InferRequest, StdTemplateInputs, TemplateInputs
from .utils import Context, ContextType, GenerationProperty, Prompt, StopWordsCriteria, Word, fetch_one, findall
from .vision_utils import load_batch, load_image, normalize_bbox, rescale_image

logger = get_logger()


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

    """

    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
    special_keys = ['images', 'videos', 'audios', 'objects']
    grounding_type = 'norm_1000'
    image_placeholder = ['<image>']
    video_placeholder = ['<video>']
    audio_placeholder = ['<audio>']
    load_medias = True

    output_prompt_answer = False  # for encoder-decoder & kto
    padding_side: Literal['left', 'right'] = 'right'  # The padding_side when the training batch_size >= 2.

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            template_meta: 'TemplateMeta',
            default_system: Optional[str] = None,
            max_length: Optional[int] = None,
            *,
            use_generate_template: bool = False,
            truncation_strategy: Literal['delete', 'truncation_left'] = 'truncation_left',
            max_pixels: Optional[int] = None,
            tools_prompt: Optional[str] = None,
            # only for train
            loss_scale: str = 'default',
            sequence_parallel_size: int = 1) -> None:
        """
        default_system: Override the default_system in the template.
        max_length: Max length of the sequence
        truncation_strategy: The truncation strategy
        loss_scale: The loss scale function to use
        max_pixels: Rescale image to reduce memory usage, default `None` means no limitation.
            e.g. 512 * 512 (H*W)
        tools_prompt: The type of tools_prompt added in the system.
        """
        from .template_meta import TemplateMeta
        if use_generate_template:
            template_meta = template_meta.to_generation_template_meta()
        # if default_system is None. not change self.default_system
        if default_system is not None:
            self.default_system = template_meta.check_system(default_system)
        else:
            self.default_system = template_meta.default_system

        template_meta.token_attr_to_id(tokenizer)

        for i, token in enumerate(template_meta.placeholder_tokens):
            if isinstance(token, str):
                template_meta.placeholder_tokens[i] = tokenizer.convert_tokens_to_ids(token)

        self.template_meta: TemplateMeta = template_meta
        self.tokenizer = tokenizer
        self.use_generate_template = use_generate_template
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.loss_scale = loss_scale
        self.max_pixels = max_pixels
        self.sequence_parallel_size = sequence_parallel_size
        self.tools_prompt = tools_prompt or template_meta.default_tools_prompt
        self.infer_backend: Literal['pt', 'vllm', 'lmdeploy'] = 'pt'
        self._handles = []
        self._deepspeed_initialize = None

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
        *,
        max_pixels: Optional[int] = None,
    ) -> None:
        template_meta = self.template_meta
        system = inputs.system
        if system is None:
            system = self.default_system
        inputs.system = template_meta.check_system(system)

        images = inputs.images
        load_medias = True if self.infer_backend in {'vllm', 'lmdeploy'} else self.load_medias
        if images and load_medias:
            images = load_batch(images, load_image)
            if max_pixels is not None:
                assert self.grounding_type != 'real', 'not support'  # TODO:check
                images = [rescale_image(img, max_pixels) for img in images]
            inputs.images = images
        if inputs.objects:
            self._preprocess_objects(inputs, inputs.objects)

        if inputs.is_multimodal:
            self._add_default_tags(inputs)

        self._get_std_messages(inputs.messages)
        n_round = len(inputs.messages) // 2
        if n_round > 1 and not template_meta.support_multi_round:
            raise ValueError(
                f'The template does not support multi-round chat, template_type: {template_meta.template_type}')

    def encode(
        self,
        inputs: Union[TemplateInputs, Dict[str, Any], StdTemplateInputs, InferRequest],
        *,
        model=None,
    ) -> Dict[str, Any]:
        """The entrance method of Template!

        Returns:
            return {'input_ids': List[int], 'labels': Optional[List[int]], ...}
        """
        if isinstance(inputs, (InferRequest, TemplateInputs)):
            # The safety is guaranteed in StdTemplateInputs.from_dict.
            inputs = dataclass_to_dict(inputs)

        if isinstance(inputs, dict):
            inputs = StdTemplateInputs.from_dict(inputs, tools_prompt=self.tools_prompt)
        elif isinstance(inputs, StdTemplateInputs):
            inputs = inputs.copy()

        assert isinstance(inputs, StdTemplateInputs)
        self._preprocess_inputs(inputs)
        if self.infer_backend in {'vllm', 'lmdeploy'}:
            res = Template._encode(self, inputs, model=model)
            if inputs.images:
                res['images'] = inputs.images
        else:
            self._check_inputs(inputs)
            res = self._encode(inputs, model=model)
        for key in ['labels', 'loss_scale']:
            if res.get(key) is None:
                res.pop(key, None)
        return res

    def post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _skip_stop_tokens(generate_ids: List[int], stop_tokens: List[int], is_finished: bool) -> List[int]:
        len_tokens = len(stop_tokens)
        if is_finished and generate_ids[-len_tokens:] == stop_tokens:
            return generate_ids[:-len_tokens]
        if not is_finished:
            for i in range(len_tokens, 0, -1):
                if generate_ids[-i:] == stop_tokens[:i]:
                    return generate_ids[:-i]
        return generate_ids

    def skip_stop_decode(self, generate_ids: List[int], is_finished: bool, **decode_kwargs) -> Any:
        # Do not print template_meta.suffix[-1] and eos_token.
        tokenizer = self.tokenizer

        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:-1]
        # skip suffix and eos_token
        template_suffix = self.template_meta.suffix[-1]
        if isinstance(template_suffix, str):
            template_suffix = tokenizer.encode(template_suffix, add_special_tokens=False)
        generate_ids = self._skip_stop_tokens(generate_ids, template_suffix, is_finished)
        return tokenizer.decode(generate_ids, **decode_kwargs)
        # if not is_finished or is_finished and response[-len_suffix:] == template_suffix:
        #     # To avoid response length being shorter than previous response length during streaming.
        #     # TODO:check
        #     # idx = max(len(response) - len_suffix, 0, self.print_idx)
        #     response = response[:-len_suffix]

    def prepare_for_generation(self,
                               generation_config,
                               inputs: Optional[Dict[str, Any]] = None,
                               model=None) -> GenerationProperty:
        return GenerationProperty(stopping_criteria=[StopWordsCriteria(self.tokenizer, generation_config.stop_words)])

    def _preprocess_objects(self, inputs: StdTemplateInputs, objects: List[Dict[str, Any]]):
        # Load image into PIL format
        images = inputs.images
        images = load_batch(images, load_image)  # base64/local_path -> PIL.Image
        # Normalize grounding bboxes
        normalize_bbox(objects, images, to_type=self.grounding_type)
        load_medias = True if self.infer_backend in {'vllm', 'lmdeploy'} else self.load_medias
        if not load_medias:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                images[i] = self._save_pil_image(image)
        inputs.images = images
        inputs.objects = objects

    @staticmethod
    def _save_pil_image(image: Image.Image) -> str:
        img_bytes = image.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        img_path = os.path.join('tmp', f'{img_hash}.png')
        if not os.path.exists(img_path):
            image.save(img_path)
        return img_path

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
                               inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
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

    def _check_inputs(self, inputs: StdTemplateInputs) -> None:
        """Check inputs valid"""
        pass

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
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
            if self.infer_backend == 'lmdeploy':
                return [[-100]]
            return self.image_placeholder
        elif media_type == 'video':
            return self.video_placeholder
        elif media_type == 'audio':
            return self.audio_placeholder

    def replace_object(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
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

    def replace_box(self, object_: Dict[str, Any], index: int, inputs: StdTemplateInputs) -> List[Context]:
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
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
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
    def _add_default_tags(inputs: StdTemplateInputs):
        total_content = '\n'.join([message['content'] for message in inputs.messages])
        for media_type in ['image', 'audio', 'video']:
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            medias = getattr(inputs, media_key)
            if not isinstance(medias, list):
                medias = [medias]
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
        if messages[0]['role'] == 'assistant':
            messages.insert(0, {'role': 'user', 'content': ''})  # pretrain
        if len(messages) % 2 == 1:
            messages.append({'role': 'assistant', 'content': None})  # inference

    def _encode(self, inputs: StdTemplateInputs, *, model: Optional[nn.Module] = None) -> Dict[str, Any]:

        res_context_list: List[Context] = []
        res_context_types: List[ContextType] = []
        template_meta = self.template_meta
        if template_meta.auto_add_bos:
            bos_token_id = self.tokenizer.bos_token_id
            if isinstance(bos_token_id, int) and bos_token_id in self.tokenizer.encode(''):
                res_context_list.append([bos_token_id])
                res_context_types.append(ContextType.OTHER)

        prefix = template_meta.system_prefix if inputs.system else template_meta.prefix
        self._concat_context_list(prefix, res_context_list, res_context_types, system=inputs.system)

        n_round = len(inputs.messages) // 2
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']
            response_role, response = response_message['role'], response_message['content']
            # TODO: Optimize the Template mechanism.
            assert query_role in {'user', 'tool'}
            assert response_role in {'assistant'}
            if query_role == 'tool':
                prompt = template_meta.tool_prompt
            elif template_meta.is_post_system and i == n_round - 1:
                prompt = template_meta.system_prompt
            else:
                prompt = template_meta.prompt

            context_list = prompt.copy()
            extra_context_list = []
            extra_context_type = None
            if i < n_round - 1:
                context_list.append('{{RESPONSE}}')
                extra_context_list = template_meta.chat_sep  # TODO:agent check
                extra_context_type = ContextType.OTHER
            elif response is not None:
                # It is the final round, and the response exists (during training).
                context_list.append('{{RESPONSE}}')
                extra_context_list = template_meta.suffix
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
            self._use_dynamic_eos(labels, self._encode_context_list(template_meta.suffix)[0])

        if tokenizer_kwargs:
            res['tokenizer_kwargs'] = tokenizer_kwargs

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
        if response is None:
            labels = None
            loss_scale = None
        if self.loss_scale == 'default':
            loss_scale = None

        res['labels'] = labels
        res['loss_scale'] = loss_scale
        return res

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        assert len(tokenizer_kwargs) == 0

    def get_generate_ids(self, generate_ids: Union[torch.Tensor, List[int]],
                         num_prompt_tokens: int) -> Union[torch.Tensor, List[int]]:
        if self.template_meta.skip_prompt:
            return generate_ids[..., num_prompt_tokens:]
        else:
            return generate_ids

    def post_process_generate_response(self, response: str, inputs: StdTemplateInputs) -> str:
        return response

    def pre_forward_hook(self,
                         args,
                         kwargs,
                         *,
                         padding_side: Optional[str] = None,
                         padding_to: Optional[int] = None,
                         model: Optional[nn.Module] = None) -> Dict[str, Any]:
        # inplace
        from swift.llm import to_device
        extra_inputs = []
        batched_data = kwargs.pop('_data')
        for data in batched_data:
            for key in ['input_ids', 'labels']:
                if key in data:
                    data[key] = torch.tensor(data[key])[None]
            extra_inputs.append(self.post_encode(model, to_device(data, model.device)))
        kwargs.update(
            to_device(
                self.data_collator(extra_inputs, padding_side=padding_side, padding_to=padding_to, model=model),
                model.device))
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        if isinstance(model, PeftModel):
            parameters = inspect.signature(model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(model.forward).parameters

        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    def set_infer_backend(self, infer_backend: Literal['vllm', 'lmdeploy', 'pt']) -> None:
        self.infer_backend = infer_backend
        if infer_backend in {'vllm', 'lmdeploy'}:
            self.remove_post_encode_hook()

    def register_post_encode_hook(self, models: List[nn.Module]) -> None:
        """This function is important for multi-modal training, as it registers the post_encode method
            as a forward hook, converting input_ids into inputs_embeds.
        """
        self.infer_backend = 'pt'
        if self._handles:
            return
        # TODO:torch>=2.0
        for model in models:
            handle = model.register_forward_pre_hook(partial(self.pre_forward_hook, model=model), with_kwargs=True)
            self._handles.append(handle)

        if is_deepspeed_zero3_enabled():
            import deepspeed
            self._deepspeed_initialize = deepspeed.initialize

            @wraps(self._deepspeed_initialize)
            def _initialize(*args, **kwargs):
                res = self._deepspeed_initialize(*args, **kwargs)
                for model, handle in zip(models, handles):
                    model._forward_pre_hooks.move_to_end(handle.id)
                return res

            deepspeed.initialize = _initialize

    def remove_post_encode_hook(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

        if self._deepspeed_initialize is not None:
            import deepspeed
            deepspeed.initialize = self._deepspeed_initialize
        self._deepspeed_initialize = None

    def pre_data_collator(self,
                          batch: List[Dict[str, Any]],
                          *,
                          padding_side: Optional[str] = None,
                          padding_to: Optional[int] = None,
                          model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """for multimodal LLM"""
        new_batch = [{'labels': b['batch'] for b in batch if b.get('labels') is not None}]
        res = self.data_collator(
            new_batch, padding_side=padding_side, padding_to=padding_to, model=model)  # only labels
        res['_data'] = batch
        return res

    def data_collator(self,
                      batch: List[Dict[str, Any]],
                      *,
                      padding_side: Optional[str] = None,
                      padding_to: Optional[int] = None,
                      model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        from swift.utils import get_dist_setting, use_torchacc
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        if padding_side is None:
            padding_side = self.padding_side
        padding_right = padding_side == 'right'
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
                res[key] = self._pad_sequence(res[key], value, padding_side)

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

        # torchacc & xtuner
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if use_torchacc():
            from swift.utils.torchacc_utils import pad_and_split_batch
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
        return res

    def print_inputs(self, inputs: Dict[str, Any], tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        for key in ['input', 'chosen_input', 'rejected_input', 'labels', 'chosen_labels', 'rejected_labels']:
            val = inputs.get(key)  # fix val is a tensor
            if val is None:
                val = inputs.get(f'{key}_ids')
            if val is not None:
                key_upper = key.upper()
                logger.info(f'[{key_upper}_IDS] {val}')
                val_str = self.safe_decode(val, **tokenizer_kwargs)
                logger.info(f'[{key_upper}] {val_str}')

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
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

    @staticmethod
    def _pad_sequence(sequences: List[torch.Tensor],
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

    def safe_decode(self, input_ids: List[int], **tokenizer_kwargs) -> str:
        placeholder_tokens = self.template_meta.placeholder_tokens

        def _is_special(token: int) -> bool:
            if token < 0:
                return True
            return token in placeholder_tokens

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if len(input_ids) == 0:
            return ''
        result_str = ''
        for i in range(len(input_ids)):
            if i == 0:
                if _is_special(input_ids[i]):
                    s = 0
                else:
                    e = 0
                continue
            if _is_special(input_ids[i]) and not _is_special(input_ids[i - 1]):
                s = i
                result_str += self.tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)
            if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
                e = i
                result_str += f'[{input_ids[i - 1]} * {e - s}]'
        if _is_special(input_ids[i]):
            result_str += f'[{input_ids[i]} * {len(input_ids) - s}]'
        else:
            result_str += self.tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
        return result_str
