# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import inspect
import os
import re
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import get_logger
from peft import PeftModel
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers.integrations import is_deepspeed_zero3_enabled

from swift.utils import dataclass_to_dict, get_dist_setting, use_torchacc
from .agent import loss_scale_map, split_str_parts_by
from .template_inputs import InferRequest, StdTemplateInputs, TemplateInputs
from .utils import (Context, ContextType, GenerationProperty, Processor, ProcessorMixin, StopWordsCriteria, fetch_one,
                    findall)
from .vision_utils import load_batch, load_image, normalize_bbox, rescale_image

logger = get_logger()


class Template(ProcessorMixin):

    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
    special_keys = ['images', 'videos', 'audios', 'objects']
    grounding_type = 'norm_1000'
    image_placeholder = ['<image>']
    video_placeholder = ['<video>']
    audio_placeholder = ['<audio>']
    load_medias = True
    skip_prompt = True

    is_encoder_decoder = False
    padding_side: Literal['left', 'right'] = 'right'  # The padding_side when the training batch_size >= 2.

    def __init__(
            self,
            processor: Processor,
            template_meta: 'TemplateMeta',
            default_system: Optional[str] = None,
            max_length: Optional[int] = None,
            *,
            use_chat_template: bool = True,
            truncation_strategy: Literal['delete', 'left'] = 'left',
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
        self.processor = processor
        self.model_info = processor.model_info
        self.model_meta = processor.model_meta
        tokenizer = self.tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        assert self.pad_token_id is not None

        if not use_chat_template:
            template_meta = template_meta.to_generate_template_meta()
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
        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.loss_scale = loss_scale
        self.max_pixels = max_pixels
        self.sequence_parallel_size = sequence_parallel_size
        self.tools_prompt = tools_prompt or template_meta.default_tools_prompt

        # infer: 'pt', 'vllm', 'lmdeploy'; train: 'train', 'rlhf', 'kto'
        self.mode: Literal['pt', 'vllm', 'lmdeploy', 'train', 'rlhf', 'kto'] = 'pt'
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
        load_medias = True if self.mode in {'vllm', 'lmdeploy'} else self.load_medias
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
            logger.warning_once(
                'The template does not support multi-round chat. Only use the last round of the conversation.')
            inputs.messages = inputs.messages[-2:]

    def _rlhf_encode(self, inputs):
        chosen_inputs, rejected_inputs = inputs, inputs.copy()
        assert chosen_inputs.rejected_response is not None, f'inputs: {inputs}'
        rejected_inputs.messages[-1]['content'] = chosen_inputs.rejected_response
        chosen_encoded = self._encode(chosen_inputs)
        rejected_encoded = self._encode(rejected_inputs)
        if len(chosen_encoded) == 0 or len(rejected_encoded) == 0:
            return {}

        encoded = {}
        for prefix in ['chosen', 'rejected']:
            data = locals()[f'{prefix}_encoded']
            for k, v in data.items():
                encoded[f'{prefix}_{k}'] = v
        return encoded

    def _kto_encode(self, inputs):
        encoded = self._rlhf_encode(inputs)
        if len(encoded) == 0:
            return {}
        return {
            'completion_input_ids': encoded['chosen_input_ids'],
            'completion_labels': encoded['chosen_labels'],
            'KL_completion_input_ids': encoded['rejected_input_ids'],
            'KL_completion_labels': encoded['rejected_labels'],
            'label': inputs.label
        }

    def encode(
        self,
        inputs: Union[TemplateInputs, Dict[str, Any], StdTemplateInputs, InferRequest],
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
        if self.mode in {'vllm', 'lmdeploy'}:
            encoded = Template._encode(self, inputs)
            if inputs.images:
                encoded['images'] = inputs.images
        elif self.mode in {'pt', 'train'}:
            encoded = self._encode(inputs)
        elif self.mode == 'rlhf':
            encoded = self._rlhf_encode(inputs)
        elif self.mode == 'kto':
            encoded = self._kto_encode(inputs)
        for key in list(encoded.keys()):
            if encoded[key] is None:
                encoded.pop(key)
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

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
        load_medias = True if self.mode in {'vllm', 'lmdeploy'} else self.load_medias
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
            if self.mode == 'lmdeploy':
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
    def _add_dynamic_eos(labels: List[int], suffix_tokens_id: List[int]) -> None:
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

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:

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
        res_context_list, loss_scale_list = loss_scale_map[self.loss_scale](res_context_list, res_context_types,
                                                                            inputs.messages)

        encoded = {}
        if self.is_encoder_decoder:
            # tokenizer_kwargs: use prompt (qwen-audio)
            answer_len = len(extra_context_list) + bool(response is not None)
            total_len = len(res_context_list)
            for key, _slice in zip(['prompt', 'answer'],
                                   [slice(0, total_len - answer_len),
                                    slice(total_len - answer_len, total_len)]):
                _res_context_list, _loss_scale_list = self._simplify_context_list(res_context_list[_slice],
                                                                                  loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                    _res_context_list, _loss_scale_list)
                encoded[f'{key}_input_ids'] = input_ids
                if key == 'answer':
                    encoded['labels'] = labels
                    encoded['loss_scale'] = loss_scale
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)
            self._add_dynamic_eos(labels, self._encode_context_list(template_meta.suffix)[0])

        if tokenizer_kwargs:
            encoded['tokenizer_kwargs'] = tokenizer_kwargs

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

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if response is None:
            for k in list(encoded.keys()):
                if k.endswith('labels'):
                    encoded[k] = None
        if response is None or self.loss_scale in {'default', 'all', 'last_round'}:
            for k in list(encoded.keys()):
                if k.endswith('loss_scale'):
                    encoded[k] = None
        return encoded

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        assert len(tokenizer_kwargs) == 0

    def get_generate_ids(self, generate_ids: Union[torch.Tensor, List[int]],
                         num_prompt_tokens: int) -> Union[torch.Tensor, List[int]]:
        if self.skip_prompt:
            return generate_ids[..., num_prompt_tokens:]
        else:
            return generate_ids

    def post_process_generate_response(self, response: str, inputs: StdTemplateInputs) -> str:
        return response

    def pre_forward_hook(self,
                         model: nn.Module,
                         args,
                         kwargs,
                         *,
                         padding_side: Optional[str] = None,
                         padding_to: Optional[int] = None) -> Dict[str, Any]:
        # inplace
        from swift.llm import to_device
        batched_data = kwargs.pop('_data')
        extra_inputs = []
        for data in batched_data:
            extra_inputs.append(self._post_encode(model, to_device(data, model.device)))
        new_kwargs = self.data_collator(extra_inputs, padding_side=padding_side, padding_to=padding_to, model=model)
        new_kwargs.pop('labels', None)
        if 'inputs_embeds' in new_kwargs:
            new_kwargs.pop('input_ids', None)
        kwargs.update(to_device(new_kwargs, model.device))

        if isinstance(model, PeftModel):
            parameters = inspect.signature(model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(model.forward).parameters

        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    def set_mode(self, mode: Literal['vllm', 'lmdeploy', 'pt', 'train', 'rlhf', 'kto']) -> None:
        self.mode = mode

    def register_post_encode_hook(self, models: List[nn.Module]) -> None:
        """This function is important for multi-modal training, as it registers the post_encode method
            as a forward hook, converting input_ids into inputs_embeds.
        """
        if self._handles:
            return

        for model in models:
            # please use torch>=2.0
            handle = model.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            self._handles.append(handle)

        if is_deepspeed_zero3_enabled():
            import deepspeed
            self._deepspeed_initialize = deepspeed.initialize

            @wraps(self._deepspeed_initialize)
            def _initialize(*args, **kwargs):
                res = self._deepspeed_initialize(*args, **kwargs)
                for model, handle in zip(models, self._handles):
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
        # compat streaming
        for data in batch:
            for key in {'input_ids', 'labels', 'loss_scale'}:
                for k, v in data.items():
                    if k.endswith(key) and isinstance(v, (list, tuple)):
                        data[k] = torch.tensor(v)[None]

        new_batch = []
        for b in batch:
            new_batch.append({k: v for k, v in b.items() if k.endswith('labels') or k == 'label'})

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
        if self.mode == 'rlhf':
            return self._rlhf_data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)
        elif self.mode == 'kto':
            return self._kto_data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)
        elif self.mode in {'pt', 'train'}:
            return self._data_collator(batch, padding_side=padding_side, padding_to=padding_to, model=model)

    @staticmethod
    def _fetch_inputs_startswith(batch: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        new_batch = []
        for inputs in batch:
            new_inputs = {}
            for k, v in inputs.items():
                if k.startswith(prefix):
                    new_inputs[k[len(prefix):]] = v
            new_batch.append(new_inputs)
        return new_batch

    def _rlhf_data_collator(self,
                            batch: List[Dict[str, Any]],
                            *,
                            chosen_prefix: str = 'chosen_',
                            rejected_prefix: str = 'rejected_',
                            padding_side: Optional[str] = None,
                            padding_to: Optional[int] = None,
                            model: Optional[nn.Module] = None) -> Dict[str, Any]:
        new_batch = []
        for prefix in [chosen_prefix, rejected_prefix]:
            new_batch += self._fetch_inputs_startswith(batch, prefix)
        return self._data_collator(new_batch, padding_side=padding_side, padding_to=padding_to, model=model)

    def _kto_data_collator(self,
                           batch: List[Dict[str, Any]],
                           *,
                           padding_side: Optional[str] = None,
                           padding_to: Optional[int] = None,
                           model: Optional[nn.Module] = None) -> Dict[str, Any]:
        kl_batch = self._fetch_inputs_startswith(batch, 'KL_completion_')
        new_batch = self._fetch_inputs_startswith(batch, 'completion_')

        kl_res = self._data_collator(kl_batch, padding_side=padding_side, padding_to=padding_to, model=model)
        res = self._data_collator(new_batch, padding_side=padding_side, padding_to=padding_to, model=model)
        if res and kl_res:
            res = {f'completion_{k}': v for k, v in res.items()}
            res.update({f'KL_completion_{k}': v for k, v in kl_res.items()})
        else:
            res = res or kl_res

        label = [b['label'] for b in batch if b.get('label') is not None]
        if label:
            res['label'] = label
        return res

    def _data_collator(self,
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
        if len(batch) == 0:
            return {}
        from swift.utils import use_torchacc
        assert self.pad_token_id is not None
        if padding_side is None:
            padding_side = self.padding_side
        padding_right = padding_side == 'right'
        res = {}
        inputs_embeds = [b['inputs_embeds'] for b in batch if b.get('inputs_embeds') is not None]
        input_ids = [b['input_ids'] for b in batch if b.get('input_ids') is not None]
        if inputs_embeds:
            res['inputs_embeds'] = inputs_embeds
        if input_ids:
            res['input_ids'] = input_ids

        for key in ['labels', 'loss_scale', 'position_ids']:
            val = [b[key] for b in batch if b.get(key) is not None]
            if val:
                res[key] = val

        keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'loss_scale', 'position_ids']
        pad_value = [self.pad_token_id, 0., 0, -100, 0., -1]
        # Convert to tensor and remove unnecessary dimensions.
        seq_lens = None
        for key in keys:
            if key not in res:
                continue
            for i, val in enumerate(res[key]):
                if isinstance(val, (list, tuple)):
                    val = torch.tensor(val)
                elif key == 'inputs_embeds' and val.ndim == 3 or key != 'inputs_embeds' and val.ndim == 2:
                    val = val[0]
                res[key][i] = val
            if not seq_lens:
                seq_lens = [seq.shape[0] for seq in res[key]]
        if seq_lens and ('input_ids' in res or 'inputs_embeds' in res):
            res['attention_mask'] = [torch.ones(seq_len, dtype=torch.int64) for seq_len in seq_lens]

        for key, pad_value in zip(keys, pad_value):
            if key not in res:
                continue
            if padding_to is not None:
                padding_len = padding_to - seq_lens[0]
                if padding_len > 0:
                    res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                        'constant', pad_value)
            res[key] = self._pad_sequence(res[key], pad_value, padding_side)

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

        if use_torchacc() or self.sequence_parallel_size > 1:
            res = self._torchacc_xtuner_data_collator(res)
        return res

    def _torchacc_xtuner_data_collator(self, res):
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
                tokenizer,
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
        for key in ['input', 'labels', 'chosen_input', 'chosen_labels', 'rejected_input', 'rejected_labels']:
            val = inputs.get(key)  # fix val is a tensor
            if val is None:
                val = inputs.get(f'{key}_ids')
            if val is not None:
                key_upper = key.upper()
                logger.info(f'[{key_upper}_IDS] {val}')
                val_str = self.safe_decode(val, **tokenizer_kwargs)
                logger.info(f'[{key_upper}] {val_str}')
        if inputs.get('loss_scale') is not None:
            val = inputs['loss_scale']
            logger.info(f'[LOSS_SCALE] {val}')

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
            if isinstance(token, float) or token < 0:
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
