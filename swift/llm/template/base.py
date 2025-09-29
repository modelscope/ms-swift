# Copyright (c) Alibaba, Inc. and its affiliates.
import hashlib
import inspect
import math
import os
import random
import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.hub.utils.utils import get_cache_dir
from peft import PeftModel
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import StoppingCriteriaList
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import strtobool

from swift.llm import to_device
from swift.utils import get_env_args, get_logger
from ..utils import Processor, ProcessorMixin
from .template_inputs import InferRequest, StdTemplateInputs, TemplateInputs
from .utils import Context, ContextType, StopWordsCriteria, fetch_one, findall, split_str_parts_by
from .vision_utils import load_audio, load_batch, load_image, rescale_image

logger = get_logger()
if TYPE_CHECKING:
    from .template_meta import TemplateMeta


class MaxLengthError(ValueError):
    pass


class Template(ProcessorMixin):
    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>', '<cot-process>', '<start-image>']
    special_keys = ['images', 'videos', 'audios', 'objects']

    image_placeholder = ['<image>']
    video_placeholder = ['<video>']
    audio_placeholder = ['<audio>']
    cot_process_placeholder = ['ки']
    placeholder_tokens = []  # For clearer printing
    load_images = True
    skip_prompt = True
    use_model = False
    norm_bbox = 'norm1000'
    support_padding_free = False  # It only takes effect for multimodal models.

    is_encoder_decoder = False

    def __init__(
        self,
        processor: Optional[Processor],
        template_meta: 'TemplateMeta',
        default_system: Optional[str] = None,
        max_length: Optional[int] = None,
        *,
        truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
        max_pixels: Optional[int] = None,
        agent_template: Optional[str] = None,
        norm_bbox: Literal['norm1000', 'none', None] = None,
        use_chat_template: bool = True,
        remove_unused_columns: bool = True,
        # only for train
        padding_free: bool = False,
        padding_side: Literal['left', 'right'] = 'right',
        loss_scale: str = 'default',
        sequence_parallel_size: int = 1,
        # infer/deploy
        response_prefix: Optional[str] = None,
        template_backend: Literal['swift', 'jinja'] = 'swift',
    ) -> None:
        """
        default_system: Override the default_system in the template.
        max_length: Max length of the sequence
        truncation_strategy: The truncation strategy
        max_pixels: Rescale image to reduce memory usage, default `None` means no limitation.
            e.g. 512 * 512 (H*W)
        padding_side: The padding_side when the training batch_size >= 2
        loss_scale: The loss scale function to use
        """
        from swift.plugin.loss_scale.loss_scale import LossScale
        from .template_meta import TemplateMeta
        from swift.plugin import agent_templates, loss_scale_map
        self._processor_inited = False
        self._version = 'v3'  # Avoid compatibility issues caused by load_from_cache_file caching.
        self.max_length = max_length
        self.model = None
        self.dummy_model = None

        if not use_chat_template:
            template_meta = template_meta.to_generate_template_meta()
        else:
            template_meta = deepcopy(template_meta)
        # if default_system is None. not change self.default_system
        template_meta.check_system(default_system)
        if default_system is not None:
            template_meta.default_system = default_system
        if response_prefix is not None:
            template_meta.response_prefix = response_prefix

        self.template_meta: TemplateMeta = template_meta
        self.use_chat_template = use_chat_template
        self.remove_unused_columns = remove_unused_columns
        self.template_backend = template_backend
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.loss_scale: LossScale = loss_scale_map[loss_scale]()
        self.max_pixels = max_pixels
        self.padding_side = padding_side
        self.sequence_parallel_size = sequence_parallel_size
        self.padding_free = padding_free  # padding_free/packing
        self.packing = False
        agent_template = agent_template or template_meta.agent_template
        self._agent_template = agent_template
        self.agent_template = agent_templates[agent_template]()
        self.norm_bbox = norm_bbox or self.norm_bbox
        if self.is_encoder_decoder:
            self.skip_prompt = False
        self.mode: Literal['pt', 'vllm', 'lmdeploy', 'sglang',  # infer
                           'train', 'rlhf', 'kto', 'gkd'] = 'pt'  # train
        self.task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'prm', 'reranker',
                                'generative_reranker'] = 'causal_lm'
        self.use_megatron = False
        self._handles = []
        self._deepspeed_initialize = None

        if processor is not None:
            self.init_processor(processor)

    def init_env_args(self):
        if self.model_meta.is_multimodal:
            self.root_image_dir = get_env_args('ROOT_IMAGE_DIR', str, None)
        else:
            self.root_image_dir = None

    def init_processor(self, processor: Processor) -> None:
        if processor is None or self._processor_inited:
            return
        self._processor_inited = True
        self.processor = processor
        self.model_info = processor.model_info
        self.config = self.model_info.config
        self.task_type = self.model_info.task_type

        self.model_meta = processor.model_meta
        if self.max_length is None:
            self.max_length = self.model_info.max_model_len
        logger.info(f'default_system: {repr(self.template_meta.default_system)}')
        logger.info(f'max_length: {self.max_length}')
        logger.info(f'response_prefix: {repr(self.template_meta.response_prefix)}')
        logger.info(f'agent_template: {self._agent_template}')
        if self.model_meta.is_multimodal:
            logger.info(f'norm_bbox: {self.norm_bbox}')
        tokenizer = self.tokenizer

        for i, token in enumerate(self.placeholder_tokens):
            if isinstance(token, str):
                self.placeholder_tokens[i] = tokenizer.convert_tokens_to_ids(token)
        self.template_meta.init(tokenizer)
        self.init_env_args()

    def _get_model(self):
        if self.model is not None:
            return self.model
        if self.dummy_model is None:
            from swift.llm import get_model_tokenizer
            with torch.device('meta'):
                self.dummy_model = get_model_tokenizer(self.model_info.model_dir, return_dummy_model=True)[0]
        return self.dummy_model

    @staticmethod
    def _load_image(image, load_images: bool):
        if load_images:
            if isinstance(image, dict) and 'bytes' in image:
                image = image['bytes'] or image['path']
            image = load_image(image)
        else:
            if isinstance(image, dict):
                path = image['path']
                if path and (path.startswith('http') or os.path.exists(path)):
                    image = path
                else:
                    image = load_image(image['bytes'])
            elif not isinstance(image, str):
                image = load_image(image)
        return image

    @staticmethod
    def _get_height_width(inputs: StdTemplateInputs) -> None:
        width = []
        height = []
        for image in inputs.images:
            width.append(image.width)
            height.append(image.height)
        inputs.objects['width'] = width
        inputs.objects['height'] = height

    def normalize_bbox(self, inputs: StdTemplateInputs) -> None:
        objects = inputs.objects
        bbox_list = objects['bbox']
        width_list = objects['width']
        height_list = objects['height']
        bbox_type = objects.pop('bbox_type', None) or 'real'
        image_id_list = objects.pop('image_id', None) or []
        image_id_list += [0] * (len(bbox_list) - len(image_id_list))
        for bbox, image_id in zip(bbox_list, image_id_list):
            if bbox_type == 'norm1':
                width, height = 1, 1
            else:
                width, height = width_list[image_id], height_list[image_id]
            for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):
                if self.norm_bbox == 'norm1000':
                    norm_width, norm_height = 1000, 1000
                elif self.norm_bbox == 'none':
                    image = inputs.images[image_id]
                    norm_width, norm_height = image.width, image.height
                bbox[2 * i] = int(round(x / width * norm_width))
                bbox[2 * i + 1] = int(round(y / height * norm_height))

    def _preprocess_function_call(self, inputs: StdTemplateInputs) -> None:
        agent_template = self.agent_template
        agent_template.template_meta = self.template_meta  # for hermes
        if inputs.tools:
            if isinstance(inputs.tools, str):
                inputs.tools = agent_template._parse_json(inputs.tools)
                if not isinstance(inputs.tools, (list, tuple)):
                    inputs.tools = [inputs.tools]
            elif isinstance(inputs.tools, (list, tuple)):
                inputs.tools = [agent_template._parse_json(tool) for tool in inputs.tools]
            else:
                raise ValueError(f'inputs.tools: {inputs.tools}')
            for i, tool in enumerate(inputs.tools):
                inputs.tools[i] = agent_template.wrap_tool(tool)
        i = 0
        messages = inputs.messages
        while i < len(messages):
            if messages[i]['role'] == 'tool_call':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool_call':
                    i += 1
                tool_content = self.agent_template._format_tool_calls(messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'assistant', 'content': tool_content}]
                i = i_start + 1
            else:
                i += 1

    def prepare_engine_kwargs(self) -> Dict[str, Any]:
        return {}

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        self._preprocess_function_call(inputs)
        if self.model_meta.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)

        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images
        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                images[i] = self._load_image(images[i], load_images)
        if inputs.objects:
            self._get_height_width(inputs)
        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            images = [rescale_image(img, self.max_pixels) for img in images]
        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images

        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))
        if inputs.is_multimodal:
            self._add_default_tags(inputs)

    @staticmethod
    def _replace_image_tags(inputs: StdTemplateInputs):
        # compat
        if inputs.images:
            return
        images = []
        pattern = r'<img>(.+?)</img>'
        for message in inputs.messages:
            content = message['content']
            if not isinstance(content, str):
                continue
            for image in re.findall(pattern, content):
                # only support local_path
                if os.path.isfile(image):
                    images.append(image)
                else:
                    logger.warning_once(f'Failed to parse image path: `{content}`.', hash_id='<img></img>')
            message['content'] = re.sub(pattern, '<image>', content)
        inputs.images = images

    @staticmethod
    def _replace_start_image_tags(inputs: StdTemplateInputs):
        # compat
        generate_mode = False
        message = inputs.messages[-1]
        content = message['content']
        if message['role'] == 'user' and content.endswith('<start-image>'):
            generate_mode = True
            message['content'] = message['content'][:-len('<start-image>')]  # remove the <start-image>
        inputs.generate_mode = generate_mode

    @staticmethod
    def _extend_tokens(
            input_ids: List[int], labels: Optional[List[int]], loss_scale: Optional[List[float]],
            replace_idx_list: List[int],
            get_new_tokens: Callable[[int], List[int]]) -> Tuple[List[int], Optional[List[int]], Optional[List[float]]]:
        added_tokens_len = 0
        for i, idx in enumerate(replace_idx_list):
            try:
                new_tokens = get_new_tokens(i)
            except IndexError as e:
                logger.warning(f'IndexError occurs in the _extend_tokens function: {e}.')
                continue
            token_len = len(new_tokens)
            input_ids = input_ids[:idx + added_tokens_len] + new_tokens + input_ids[added_tokens_len + idx + 1:]
            if labels:
                labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx + 1:]
            if loss_scale:
                scale_idx = loss_scale[idx + added_tokens_len]
                loss_scale = loss_scale[:idx + added_tokens_len] + [scale_idx] * token_len + loss_scale[added_tokens_len
                                                                                                        + idx + 1:]
            added_tokens_len += token_len - 1
        return input_ids, labels, loss_scale

    def forward_context(self, model, inputs):
        # This function is only used to handle scenarios where the model needs
        # to be patched during the forward pass.
        return nullcontext()

    @staticmethod
    def get_base_model(model):
        if isinstance(model, PeftModel):
            return model.model
        else:
            return model

    def _rlhf_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        chosen = inputs.chosen
        margin = chosen.margin
        chosen_encoded = self._encode_truncated(chosen)
        rejected_encoded = self._encode_truncated(inputs.rejected)

        encoded = {}
        for prefix in ['chosen', 'rejected']:
            data = locals()[f'{prefix}_encoded']
            for k, v in data.items():
                encoded[f'{prefix}_{k}'] = v
        if margin is not None:
            encoded['margin'] = float(margin)
        return encoded

    def _kto_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        encoded = self._rlhf_encode(inputs)
        encoded['label'] = bool(inputs.chosen.label)
        return encoded

    def _gkd_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = self._encode_truncated(inputs)
        encoded['prompts'] = encoded['input_ids'][:-len(encoded.pop('answer_input_ids'))]
        for k in list(encoded.keys()):
            if k.startswith('prompt_') or k.endswith('answer_'):
                encoded.pop(k, None)
        return encoded

    def _embedding_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        _encoded = {}
        labels = []

        if self.is_training:
            anchor = inputs.chosen
            anchor_encoded = self._encode_truncated(anchor)
            for key in anchor_encoded:
                _encoded[f'anchor_{key}'] = anchor_encoded[key]
            positive = inputs.positive
            if isinstance(positive, list):
                positive = positive[0]
            positive_encoded = self._encode_truncated(positive)
            for key in positive_encoded:
                _encoded[f'positive_{key}'] = positive_encoded[key]
            labels.append(float(inputs.chosen.label) if inputs.chosen.label is not None else 1.0)

            _all_negative_keys = set()
            for idx, negative in enumerate(inputs.negative):
                _tmp_negative_keys = set()
                negative_encoded = self._encode_truncated(negative)
                for key in negative_encoded:
                    negative_key = f'negative_{key}'
                    _all_negative_keys.add(negative_key)
                    _tmp_negative_keys.add(negative_key)
                    if negative_key not in _encoded:
                        _encoded[negative_key] = [None] * idx
                    _encoded[negative_key].append(negative_encoded[key])
                for miss_key in (_all_negative_keys - _tmp_negative_keys):
                    _encoded[miss_key].append(None)
                labels.append(0.0)

            _encoded['labels'] = labels
        else:
            anchor = inputs.chosen
            _encoded = self._encode_truncated(anchor)
            _encoded.pop('labels', None)
        return _encoded

    def _reranker_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
        if self.is_training:
            chosen = inputs.chosen
            instruction = chosen.system

            _encoded = defaultdict(list)
            labels = []

            for positive in inputs.positive:
                if instruction is not None and positive.system is None:
                    positive.system = instruction
                positive.messages = chosen.messages + positive.messages
                positive_encoded = self._encode_truncated(positive)
                labels.append(1)
                for key in positive_encoded:
                    _encoded[key].append(positive_encoded[key])

            for negative in inputs.negative:
                if instruction is not None and negative.system is None:
                    negative.system = instruction
                negative.messages = chosen.messages + negative.messages
                negative_encoded = self._encode_truncated(negative)
                labels.append(0)
                for key in negative_encoded:
                    _encoded[key].append(negative_encoded[key])

            _encoded['labels'] = labels
        else:
            anchor = inputs.chosen
            _encoded = self._encode_truncated(anchor)
            _encoded.pop('labels', None)
        return _encoded

    def _seq_cls_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = self._encode_truncated(inputs)
        encoded.pop('labels', None)
        if inputs.label is not None:
            labels = inputs.label
            problem_type = self.config.problem_type
            if problem_type == 'single_label_classification':
                labels = int(labels)
            encoded['labels'] = labels
        return encoded

    @torch.inference_mode()
    def encode(self,
               inputs: Union[TemplateInputs, Dict[str, Any], InferRequest],
               return_template_inputs: bool = False,
               return_length: bool = False) -> Dict[str, Any]:
        """The entrance method of Template!

        Returns:
            return {'input_ids': List[int], 'labels': Optional[List[int]], ...}
        """
        assert self._processor_inited, ('Please initialize the processor before calling the template.encode method: '
                                        'template.init_processor(processor).')
        if isinstance(inputs, InferRequest):
            inputs = asdict(inputs)

        if isinstance(inputs, dict):
            if self.task_type == 'causal_lm' and not self.is_training:
                InferRequest.remove_response(inputs['messages'])
            inputs = TemplateInputs.from_dict(inputs)
        elif isinstance(inputs, TemplateInputs):
            inputs = deepcopy(inputs)
        assert isinstance(inputs, TemplateInputs)

        chosen = inputs.chosen
        if self.task_type == 'causal_lm':
            if self.mode in {'train', 'pt', 'vllm', 'lmdeploy', 'sglang'}:
                encoded = self._encode_truncated(chosen)
            elif self.mode == 'rlhf':
                encoded = self._rlhf_encode(inputs)
            elif self.mode == 'kto':
                encoded = self._kto_encode(inputs)
            elif self.mode == 'gkd':
                encoded = self._gkd_encode(chosen)
        elif self.task_type == 'seq_cls':
            if self.mode == 'rlhf':
                encoded = self._rlhf_encode(inputs)
                for prefix in ['chosen', 'rejected']:
                    encoded.pop(f'{prefix}_labels', None)
                    encoded.pop(f'{prefix}_loss_scale', None)
            else:
                encoded = self._seq_cls_encode(chosen)
        elif self.task_type == 'prm':
            encoded = self._encode_truncated(chosen)
        elif self.task_type == 'embedding':
            encoded = self._embedding_encode(inputs)
        elif self.task_type in {'reranker', 'generative_reranker'}:
            encoded = self._reranker_encode(inputs)
        else:
            raise ValueError(f'task_type: {self.task_type} is not supported.')

        if chosen.channel is not None:
            encoded['channel'] = chosen.channel

        lengths = [0] if self.task_type not in {'reranker', 'generative_reranker'} else []
        for key in list(encoded.keys()):
            if encoded[key] is None:
                encoded.pop(key)
            elif key.endswith('length'):
                value = encoded[key]
                if isinstance(value, int):
                    lengths.append(value)
                elif isinstance(value, (tuple, list)):
                    lengths += value
        if return_length:
            if self.task_type in {'reranker', 'generative_reranker'}:
                encoded['length'] = lengths
            else:
                encoded['length'] = sum(lengths)
        else:
            encoded.pop('length', None)
        if return_template_inputs:
            encoded['template_inputs'] = chosen
        if not self.remove_unused_columns:
            encoded['_extra_kwargs'] = chosen.extra_kwargs
        return encoded

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        packed = {}
        keys = set()
        length = []
        for r in row:
            keys.update(r.keys())
            length.append(r['length'])
        for key in keys:
            if key in {'input_ids', 'labels', 'loss_scale'}:
                packed[key] = sum((x[key] for x in row), start=[])
            elif key == 'length':
                packed[key] = sum((x[key] for x in row))
            elif key == 'channel':
                packed[key] = [x.get(key) for x in row]
        if 'position_ids' not in packed:
            packed['position_ids'] = sum((list(range(x)) for x in length), start=[])

        packed.update(self._data_collator_mm_data(row))
        return packed

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    @staticmethod
    def _get_seq_cls_logprobs(pred: int, logprobs: torch.Tensor, top_logprobs: int):
        idxs = logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
        logprobs = logprobs.tolist()
        return {
            'content': [{
                'index': pred,
                'logprobs': [logprobs[p] for p in pred] if isinstance(pred, (list, tuple)) else logprobs[pred],
                'top_logprobs': [{
                    'index': idx,
                    'logprob': logprobs[idx]
                } for idx in idxs]
            }]
        }

    def decode_seq_cls(self, logits: torch.Tensor, top_logprobs: int):
        assert isinstance(logits, torch.Tensor)
        problem_type = self.config.problem_type
        if problem_type == 'regression':
            preds = logits.squeeze(dim=-1).tolist()
            logprobs = [None] * len(preds)
        else:
            if problem_type == 'single_label_classification':
                preds = torch.argmax(logits, dim=-1).tolist()
                logprobs = torch.log_softmax(logits, -1)
            else:
                preds = [(logprob >= 0.5).nonzero(as_tuple=True)[0].tolist() for logprob in torch.sigmoid(logits)]
                logprobs = F.logsigmoid(logits)
            logprobs = [self._get_seq_cls_logprobs(pred, logprobs[i], top_logprobs) for i, pred in enumerate(preds)]
        return preds, logprobs

    def decode(self, generate_ids: List[int], *, is_finished: bool = True, first_token=True, **kwargs) -> Any:
        if kwargs.get('spaces_between_special_tokens') is None:
            kwargs['spaces_between_special_tokens'] = False
        generate_ids = self.skip_stop_tokens(generate_ids, is_finished)
        response = self.tokenizer.decode(generate_ids, **kwargs)
        if first_token and self.template_meta.response_prefix:
            response = self.template_meta.response_prefix + response
        return response

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        raise NotImplementedError

    @contextmanager
    def generate_context(self):
        origin_mode = self.mode
        if self.mode in {'train', 'rlhf', 'kto', 'gkd'}:
            self.set_mode('pt')
        is_multimodal = self.model_meta.is_multimodal
        if is_multimodal:
            models = self.remove_post_encode_hook()
        try:
            yield
        finally:
            if is_multimodal:
                self.register_post_encode_hook(models)
            self.set_mode(origin_mode)

    def generate(self, model, *args, **kwargs):
        base_model = self.get_base_model(model)
        signature = inspect.signature(base_model.generate)
        if 'use_model_defaults' in signature.parameters and 'use_model_defaults' not in kwargs:
            kwargs['use_model_defaults'] = False
        return model.generate(*args, **kwargs)

    def skip_stop_tokens(self, generate_ids: List[int], is_finished: bool = True) -> List[int]:
        # Do not print template_meta.suffix[-1] and eos_token.
        # However, other stop_words will be printed.
        tokenizer = self.tokenizer

        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:-1]
        # skip suffix and eos_token
        template_suffix = self.template_meta.suffix[-1]
        if isinstance(template_suffix, str):
            # [-1:]: fix OpenGVLab/Mini-InternVL-Chat-4B-V1-5
            template_suffix = tokenizer.encode(template_suffix, add_special_tokens=False)[-1:]

        len_tokens = len(template_suffix)
        if is_finished and generate_ids[-len_tokens:] == template_suffix:
            generate_ids = generate_ids[:-len_tokens]
        elif not is_finished:
            for i in range(len_tokens, 0, -1):
                if generate_ids[-i:] == template_suffix[:i]:
                    generate_ids = generate_ids[:-i]
                    break
        return generate_ids

    def prepare_generate_kwargs(self, generate_kwargs: Dict[str, Any], *, model=None) -> Dict[str, Any]:
        generation_config = generate_kwargs['generation_config']
        stop_words = getattr(generation_config, 'stop_words', None) or self.template_meta.stop_words
        generate_kwargs['stopping_criteria'] = StoppingCriteriaList([StopWordsCriteria(self.tokenizer, stop_words)])
        return generate_kwargs

    @staticmethod
    def _save_pil_image(image: Image.Image) -> str:
        img_bytes = image.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        tmp_dir = os.path.join(get_cache_dir(), 'tmp', 'images')
        logger.info_once(f'create tmp_dir: {tmp_dir}')
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, f'{img_hash}.png')
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

    def _tokenize(self, context, **kwargs):
        return self.tokenizer(context, return_attention_mask=False, add_special_tokens=False, **kwargs)['input_ids']

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
            if self.mode == 'vllm':
                # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
                from vllm.assets.video import video_to_ndarrays, video_get_metadata
                num_frames = get_env_args('vllm_num_frames', int, 16)
                video_data = video_to_ndarrays(inputs.videos[index], num_frames)
                video_metadatas = video_get_metadata(inputs.videos[index], num_frames)
                inputs.videos[index] = [(video_data, video_metadatas)]
                return self.video_placeholder
            else:
                return self.video_placeholder
        elif media_type == 'audio':
            return self.audio_placeholder

    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace objects referenced by the bbox to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            ref: Description of the bbox
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [ref]

    def replace_cot_process(self, inputs: StdTemplateInputs) -> List[Context]:
        """Replace the cot process label for PRM training or inference.
        Override this function to do your own replace operation.

        Args:
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [self.cot_process_placeholder]

    @staticmethod
    def _get_bbox_str(bbox: List[int]) -> str:
        point = []
        for x, y in zip(bbox[::2], bbox[1::2]):
            point.append(f'({x},{y})')
        return ','.join(point)

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            bbox: [x, y] or [x1, y1, x2, y2]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [f'[{self._get_bbox_str(bbox)}]']

    def _pre_tokenize_images(self, context_list: List[Context], loss_scale_list: List[float],
                             inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        # https://github.com/modelscope/ms-swift/issues/3407
        # Fix the bounding box position offset issue in the Qwen2.5-VL grounding task.
        res: List[Context] = []
        res_loss_scale: List[float] = []
        inputs.image_idx = 0

        for context, loss_scale in zip(context_list, loss_scale_list):
            if context == '<image>' and inputs.is_multimodal and inputs.image_idx < len(inputs.images):
                c_list = self.replace_tag('image', inputs.image_idx, inputs)
                inputs.image_idx += 1
                loss_scale = 0. if self.template_backend == 'swift' else 1.
            else:
                c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

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
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)
        # replace tag/object/box
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        # reset
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['video', 'audio']:
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(
                        getattr(inputs, f'{k}s')):
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    loss_scale = 0.
                    break
            else:
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox):
                    idx = inputs.bbox_idx
                    c_list = self.replace_bbox(bbox[idx], idx, inputs)
                    inputs.bbox_idx += 1
                elif context == '<cot-process>' and self.task_type == 'prm':
                    c_list = self.replace_cot_process(inputs)
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    @staticmethod
    def _add_default_tags(inputs: StdTemplateInputs):
        total_content = []
        for message in inputs.messages:
            content = message['content'] or ''
            if not isinstance(content, str):
                if message['role'] == 'user':
                    # Give up adding the default tag
                    return
                elif message['role'] == 'assistant':
                    continue
            total_content.append(content)
        total_content = '\n'.join(total_content)
        if inputs.system:
            total_content = f'{inputs.system}\n{total_content}'
        for media_type in ['image', 'audio', 'video']:
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            medias = getattr(inputs, media_key)
            if not isinstance(medias, list):
                medias = [medias]
            if medias:
                num_media_tags = len(re.findall(media_tag, total_content))
                num_media = len(medias)
                num_new_tags = num_media - num_media_tags
                if num_new_tags > 0:
                    inputs.messages[0]['content'] = media_tag * num_new_tags + inputs.messages[0]['content']
                elif num_new_tags < 0:
                    logger.warning(
                        f'num_media: {num_media}, num_media_tags: {num_media_tags}, total_content: {total_content}. '
                        'We will only replace the frontmost media_tags while keeping the subsequent media_tags.')

    def _encode_context_list(self,
                             context_list: List[Context],
                             loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float]]:
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                token_list = self._tokenize(context)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            if not self.loss_scale.is_loss_scale_binary:
                loss_scale.extend([loss_weight] * len(token_list))
        if self.loss_scale.is_loss_scale_binary:
            loss_scale = None
        return input_ids, labels, loss_scale

    @staticmethod
    def _add_dynamic_eos(input_ids: List[int], labels: List[int], loss_scale: Optional[List[int]],
                         suffix_tokens_id: List[int]) -> None:
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels)):
            if labels[i - 1] >= 0 and labels[i] == -100:
                start = i
            if start > 0 and labels[i - 1] == -100 and labels[i] >= 0:
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len and input_ids[start:start + suffix_len] == suffix_tokens_id:
                    labels[start:start + suffix_len] = suffix_tokens_id
                    if loss_scale and loss_scale[start:start + suffix_len] == [0] * suffix_len:
                        loss_scale[start:start + suffix_len] = [1] * suffix_len

    @staticmethod
    def _get_std_messages(messages):
        if messages and messages[0]['role'] == 'assistant':
            messages.insert(0, {'role': 'user', 'content': ''})  # pretrain
        if len(messages) % 2 == 1:
            messages.append({'role': 'assistant', 'content': None})  # inference

    def _jinja_encode(self, inputs: StdTemplateInputs):
        messages = inputs.messages.copy()
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        if messages[-1]['content'] is None:
            messages.pop()
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        kwargs = {}
        if inputs.tools:
            kwargs['tools'] = inputs.tools
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        answer_len = 1 if self.is_training else 0
        return [text], [1.], answer_len

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        template_meta = self.template_meta
        system = inputs.system
        tools = inputs.tools
        template_meta.check_system(system)
        if system is None:
            system = template_meta.default_system

        if tools is not None:
            system = self.agent_template._format_tools(tools, system, inputs.messages[0])
        return system

    def _swift_prepare_inputs(self, inputs: StdTemplateInputs):
        """
        Preprocesses the list of messages in the input by merging and formatting consecutive messages
        according to their roles.

        Specifically, this method:
            - Merges consecutive messages from the same role ('assistant' or 'user') to prevent downstream errors.
            - Detects consecutive tool-related messages following an assistant message, then formats and
            combines them using `agent_template._format_tool_responses` for structured output.
            - Updates the messages list in-place for further processing.

        Args:
            inputs: An StdTemplateInputs object which contains a 'messages' attribute, which is a list of dictionaries.
                    Each message dictionary should have at least the keys 'role' and 'content'.

        Returns:
            None. The input messages list is updated in-place.
        """
        messages = inputs.messages
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                # where tool_content is a List.
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif pre_role == 'assistant' and role == 'assistant' or pre_role == 'user' and role == 'user':
                # Consecutive messages from the assistant/user role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    def _swift_encode(self, inputs: StdTemplateInputs):
        template_meta = self.template_meta
        self._swift_prepare_inputs(inputs)
        system = self._get_system(inputs)

        self._get_std_messages(inputs.messages)
        n_round = len(inputs.messages) // 2
        if n_round > 1 and not self.template_meta.support_multi_round:
            logger.warning_once(
                'The template does not support multi-round chat. Only use the last round of the conversation.')
            # TODO: Multimodal models may encounter image mismatch issues.
            inputs.messages = inputs.messages[-2:]

        res_context_list: List[Context] = []
        res_context_types: List[ContextType] = []
        sep_token = None
        if template_meta.auto_add_bos:
            all_tokens = self.tokenizer.encode('a')
            single_token = self.tokenizer.encode('a', add_special_tokens=False)
            assert len(single_token) == 1
            idx = all_tokens.index(single_token[0])
            bos_token = all_tokens[:idx]
            sep_token = all_tokens[idx + 1:]
            if bos_token:
                res_context_list.append(bos_token)
                res_context_types.append(ContextType.OTHER)

        if self.template_meta.is_post_system or not system:
            prefix = template_meta.prefix
        else:
            prefix = template_meta.system_prefix
        self._concat_context_list(prefix, res_context_list, res_context_types, system=system)

        assert len(inputs.messages) > 0, f'inputs.messages: {inputs.messages}'
        n_round = len(inputs.messages) // 2
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']
            response_role, response = response_message['role'], response_message['content']
            # TODO: Optimize the Template mechanism.
            assert query_role in {'user', 'tool'}, f'query_role: "{query_role}"'
            assert response_role in {'assistant'}, f'response_role: "{response_role}"'
            if query_role == 'tool':
                prompt = query
                query = ''
            elif template_meta.is_post_system and i == n_round - 1:
                prompt = template_meta.system_prompt
            else:
                prompt = template_meta.prompt

            context_list = prompt.copy()
            extra_context_list = []
            extra_context_type = None
            if i < n_round - 1:
                # Not the last round.
                context_list.append('{{RESPONSE}}')
                if inputs.messages[2 * (i + 1)]['role'] != 'tool':
                    extra_context_list = template_meta.chat_sep
                    extra_context_type = ContextType.OTHER
            elif response is not None:
                # It is the final round, and the response exists (during training).
                context_list.append('{{RESPONSE}}')
                # The GLM-4.5 assistant part (tool call) may end with <|observation|>,
                # and here we avoid adding <|user|>.
                response_content = response
                if not isinstance(response_content, str):
                    if isinstance(response, list):
                        token_ids = response
                    else:
                        token_ids = response['token_ids']
                    response_content = self.tokenizer.decode(token_ids[-20:])
                endswith_stop_words = any(
                    response_content.endswith(stop_word) for stop_word in template_meta.stop_words
                    if isinstance(stop_word, str))
                # self.is_training needed because we may want to continue generation from
                # the current response
                if (self.is_training or self.task_type != 'causal_lm') and not sep_token and not endswith_stop_words:
                    extra_context_list = template_meta.suffix
                    extra_context_type = ContextType.SUFFIX
            elif template_meta.response_prefix:
                # final round and during inference.
                context_list.append(template_meta.response_prefix)

            self._concat_context_list(
                context_list,
                res_context_list,
                res_context_types,
                query=query,
                response=response,
                system=system,
                round0=i)
            res_context_list += extra_context_list
            res_context_types += [extra_context_type] * len(extra_context_list)
        if template_meta.auto_add_bos and sep_token:
            res_context_list.append(sep_token)
            res_context_types.append(ContextType.SUFFIX)
        res_context_list, loss_scale_list = self.loss_scale(res_context_list, res_context_types, inputs.messages)
        if self.is_training:
            answer_len = len(extra_context_list) + bool(response is not None)
        else:
            answer_len = 0
        return res_context_list, loss_scale_list, answer_len

    def _truncate(self, input_ids: List[int], labels: Optional[List[int]], loss_mask: Optional[List[float]],
                  truncation_strategy: Literal['left', 'right']):
        placeholder_tokens = torch.tensor(self.placeholder_tokens)
        input_ids_tensor = torch.tensor(input_ids)
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
        n_protected = protected.sum().item()
        if n_protected < self.max_length:
            non_protected = (~protected).nonzero(as_tuple=True)[0]
            if truncation_strategy == 'left':
                idx = non_protected[-(self.max_length - n_protected):]
            else:
                idx = non_protected[:self.max_length - n_protected]
            protected[idx] = True
        input_ids = input_ids_tensor[protected].tolist()
        if labels is not None:
            labels = torch.tensor(labels)[protected].tolist()
        if loss_mask is not None:
            loss_mask = torch.tensor(loss_mask)[protected].tolist()
        return input_ids, labels, loss_mask

    @staticmethod
    def _get_length(input_ids, labels):
        # input_ids might be a tensor.
        lengths = [0]
        if input_ids is not None:
            lengths.append(len(input_ids))
        if labels is not None:
            lengths.append(len(labels))
        length = max(lengths)
        return length

    def _encode_truncated(self, inputs: StdTemplateInputs):
        self._preprocess_inputs(inputs)
        if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
            # For multi-modal models, images do not need to be pre processed here
            # vllm/lmdeploy/sglang will handle the logic
            encoded = Template._encode(self, inputs)
            keys = ['images', 'audios', 'videos']
            if self.mode == 'vllm':
                keys.append('mm_processor_kwargs')
            for key in keys:
                value = getattr(inputs, key)
                if value:
                    encoded[key] = value
        else:
            encoded = self._encode(inputs)
        self._handle_megatron_cp(encoded)  # TODO: fix cp_size & cached_dataset
        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        loss_scale = encoded.get('loss_scale')
        length = self._get_length(input_ids, labels)
        if self.max_length is not None and length > self.max_length:
            if self.truncation_strategy in {'right', 'left'}:
                input_ids, labels, loss_scale = self._truncate(
                    input_ids, labels, loss_scale, truncation_strategy=self.truncation_strategy)
                length = self._get_length(input_ids, labels)
            elif self.truncation_strategy == 'raise':
                raise MaxLengthError(f'Current length of row({length}) is larger'
                                     f' than the max_length({self.max_length}).')
        encoded['length'] = length
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        inputs.messages = deepcopy(inputs.messages)
        template_backend = self.template_backend
        if (self.template_meta.template_type == 'dummy' and self.use_chat_template and not self.is_training
                and self.task_type == 'causal_lm'):
            template_backend = 'jinja'
            logger.info_once(f'Setting template_backend: {template_backend}')
        res_context_list, loss_scale_list, answer_len = (
            self._swift_encode(inputs) if template_backend == 'swift' else self._jinja_encode(inputs))
        encoded = {}
        if self.is_encoder_decoder or self.mode == 'gkd':
            total_len = len(res_context_list)
            for key, _slice in zip(['prompt', 'answer'],
                                   [slice(0, total_len - answer_len),
                                    slice(total_len - answer_len, total_len)]):
                context_list, loss_scale = self._simplify_context_list(res_context_list[_slice],
                                                                       loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale = self._encode_context_list(context_list, loss_scale)
                encoded[f'{key}_input_ids'] = input_ids
                encoded[f'{key}_labels'] = labels
                encoded[f'{key}_loss_scale'] = loss_scale
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
            labels = encoded['prompt_labels'] + encoded['answer_labels']
            loss_scale = None
            if isinstance(encoded['prompt_loss_scale'], list):
                loss_scale = encoded['prompt_loss_scale'] + encoded['answer_loss_scale']
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale = self._encode_context_list(res_context_list, loss_scale_list)
        self._add_dynamic_eos(input_ids, labels, loss_scale, self._encode_context_list(self.template_meta.suffix)[0])

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if encoded.get('labels') is not None:
            encoded['labels'][0] = -100
        if encoded.get('loss_scale') is not None:
            encoded['loss_scale'][0] = 0
        if not self.is_training:
            for k in list(encoded.keys()):
                if k.endswith('labels') or k.endswith('loss_scale'):
                    encoded[k] = None
        return encoded

    def _handle_megatron_cp(self, encoded: Dict[str, Any]) -> None:
        cp_size = self.sequence_parallel_size
        if not self.use_megatron or cp_size == 1:
            return
        input_ids = encoded['input_ids']
        padding_len = math.ceil(len(input_ids) / (cp_size * 2)) * (cp_size * 2) - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        encoded['labels'] += [-100] * padding_len
        if encoded.get('loss_scale') is not None:
            encoded['loss_scale'] += [0] * padding_len

    def debug_logger(self, inputs):
        if not strtobool(os.getenv('SWIFT_DEBUG', 'false')):
            return
        if 'input_ids' in inputs:
            k = 'input_ids'
            val = inputs['input_ids']
        elif 'generate_ids' in inputs:
            k = 'generate_ids'
            val = inputs['generate_ids']
        for v in val:
            self.print_inputs({k: v.tolist()})

    @staticmethod
    def _split_list(inputs: List[int], x: int) -> List[List[int]]:
        idxs = findall(inputs, x)
        idxs.append(len(inputs))
        res = []
        lo = 0
        for idx in idxs:
            res.append(inputs[lo:idx])
            lo = idx + 1
        return res

    def replace_video2image(self, load_video_func, inputs, replace_tag: Callable) -> List[Context]:
        context_list = []
        if self.mode in {'vllm', 'lmdeploy'}:
            video = inputs.videos.pop(inputs.video_idx)
            inputs.video_idx -= 1
        else:
            video = inputs.videos[inputs.video_idx]
        images = inputs.images
        new_images = load_video_func(video)
        inputs.images = images[:inputs.image_idx] + new_images + images[inputs.image_idx:]
        for i in range(len(new_images)):
            context_list += replace_tag(i)
        inputs.image_idx += len(new_images)
        return context_list

    def get_generate_ids(self, generate_ids: Union[torch.Tensor, List[int]],
                         num_prompt_tokens: int) -> Union[torch.Tensor, List[int]]:
        if self.skip_prompt:
            generate_ids = generate_ids[..., num_prompt_tokens:]
        return generate_ids

    def post_process_generate_response(self, response: str, inputs: StdTemplateInputs) -> str:
        return response

    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        old_kwargs = to_device(kwargs, model.device)
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        for k, v in old_kwargs.items():
            if k in {
                    'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                    'max_length_q', 'max_length_k', 'cu_seq_lens_q', 'cu_seq_lens_k'
            } and k not in kwargs:
                kwargs[k] = v
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        base_model = self.get_base_model(model)
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    @property
    def is_training(self):
        return self.mode not in {'pt', 'vllm', 'lmdeploy', 'sglang'}

    def set_mode(self, mode: Literal['pt', 'vllm', 'lmdeploy', 'sglang', 'train', 'rlhf', 'kto', 'gkd']) -> None:
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
            self._handles.append((model, handle))

        if is_deepspeed_zero3_enabled():
            import deepspeed
            self._deepspeed_initialize = deepspeed.initialize

            @wraps(self._deepspeed_initialize)
            def _initialize(*args, **kwargs):
                res = self._deepspeed_initialize(*args, **kwargs)
                for model, handle in self._handles:
                    model._forward_pre_hooks.move_to_end(handle.id)
                return res

            deepspeed.initialize = _initialize

    def remove_post_encode_hook(self):
        models = []
        for model, handle in self._handles:
            models.append(model)
            handle.remove()
        self._handles = []

        if self._deepspeed_initialize is not None:
            import deepspeed
            deepspeed.initialize = self._deepspeed_initialize
        self._deepspeed_initialize = None
        return models

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        from swift.llm import RowPreprocessor
        if self.packing and isinstance(batch[0], list):
            batch = sum(batch, start=[])
        num_samples = len(batch)
        if self.task_type == 'causal_lm':
            if self.mode in {'pt', 'train'}:
                res = self._data_collator(batch, padding_to=padding_to)
            elif self.mode == 'rlhf':
                res = self._rlhf_data_collator(batch, padding_to=padding_to)
            elif self.mode == 'kto':
                res = self._kto_data_collator(batch, padding_to=padding_to)
            elif self.mode == 'gkd':
                res = self._gkd_data_collator(batch, padding_to=padding_to)
        elif self.task_type == 'prm':
            res = self._data_collator(batch, padding_to=padding_to)
        elif self.task_type == 'seq_cls':
            if self.mode == 'rlhf':
                res = self._rlhf_data_collator(batch, padding_to=padding_to)
            else:
                res = self._seq_cls_data_collator(batch, padding_to=padding_to)
        elif self.task_type == 'embedding':
            res = self._embedding_data_collator(batch, padding_to=padding_to)
        elif self.task_type in {'reranker', 'generative_reranker'}:
            res = self._reranker_data_collator(batch, padding_to=padding_to)
        else:
            raise ValueError(f'task_type: {self.task_type} is not supported.')
        if not self.remove_unused_columns:
            extra_kwargs = [b['_extra_kwargs'] for b in batch if b.get('_extra_kwargs') is not None]
            extra_kwargs = RowPreprocessor.rows_to_batched(extra_kwargs)
            res.update({k: v for k, v in extra_kwargs.items() if k not in res})
        if self.use_megatron:
            res['num_samples'] = num_samples
        return res

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

    @staticmethod
    def fetch_inputs(batch: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, Any]:
        from swift.llm import RowPreprocessor
        keys = keys or []
        rows = RowPreprocessor.rows_to_batched(batch)
        return {k: rows[k] for k in keys if rows.get(k) is not None}

    @staticmethod
    def gather_list(batch: List[Dict[str, Any]], attr_name: str) -> Optional[List[Any]]:
        # List[Tensor] ->  List[Tensor]
        res = []
        for b in batch:
            if b.get(attr_name) is not None:
                res += b.pop(attr_name)
        return res

    @staticmethod
    def concat_tensor(batch: List[Dict[str, Any]], attr_name: str, dim: int) -> Optional[torch.Tensor]:
        res = []
        for b in batch:
            if b.get(attr_name) is not None:
                res.append(b.pop(attr_name))
        return torch.concat(res, dim=dim) if res else None

    def _rlhf_data_collator(self,
                            batch: List[Dict[str, Any]],
                            *,
                            chosen_prefix: str = 'chosen_',
                            rejected_prefix: str = 'rejected_',
                            padding_to: Optional[int] = None) -> Dict[str, Any]:
        new_batch = []
        for prefix in [chosen_prefix, rejected_prefix]:
            new_batch += self._fetch_inputs_startswith(batch, prefix)
        res = self._data_collator(new_batch, padding_to=padding_to)

        # reward modeling
        margin = [b['margin'] for b in batch if b.get('margin') is not None]
        if margin:
            res['margin'] = torch.tensor(margin, dtype=torch.float)

        return res

    def _kto_data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        new_batch = self._fetch_inputs_startswith(batch, 'chosen_')
        kl_batch = self._fetch_inputs_startswith(batch, 'rejected_')

        res = self._data_collator(new_batch, padding_to=padding_to)
        kl_res = self._data_collator(kl_batch, padding_to=padding_to)
        res = {
            **{f'completion_{k}': v
               for k, v in res.items()},
            **{f'KL_completion_{k}': v
               for k, v in kl_res.items()},
        }
        label = [b['label'] for b in batch if b.get('label') is not None]
        if label:
            res['label'] = label
        return res

    def _gkd_data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = self._data_collator(batch, padding_to=padding_to)
        prompts_batch = [{'input_ids': b['prompts']} for b in batch if b.get('prompts') is not None]
        if prompts_batch:
            prompts_res = self._data_collator(prompts_batch, padding_to=padding_to)
            res['prompts'] = prompts_res.pop('input_ids')
            res.update({f'prompt_{k}': v for k, v in prompts_res.items()})
        return res

    def _embedding_data_collator(self,
                                 batch: List[Dict[str, Any]],
                                 *,
                                 padding_to: Optional[int] = None) -> Dict[str, Any]:
        labels = []
        new_batch = []
        for b in batch:
            if 'input_ids' in b:
                new_batch += [b]
            else:
                keys = [key for key in b.keys() if 'negative' in key]
                max_neg = None
                for key in keys:
                    value_list = b[key]
                    suffix = key[len('negative_'):]
                    max_neg = len(value_list)
                    for i, value in enumerate(value_list):
                        b[f'negative{i}_{suffix}'] = value
                    b.pop(key)

                indexes = ['anchor_', 'positive_']
                if max_neg is not None:
                    for i in range(0, max_neg):
                        indexes.append(f'negative{i}_')
                for prefix in indexes:
                    new_batch += self._fetch_inputs_startswith([b], prefix)
            labels.extend(b.get('labels', []))
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels:
            res['labels'] = torch.tensor(labels, dtype=torch.float32)
        return res

    def _reranker_data_collator(self,
                                batch: List[Dict[str, Any]],
                                *,
                                padding_to: Optional[int] = None) -> Dict[str, Any]:
        if self.is_training:
            max_positive_samples = int(os.environ.get('MAX_POSITIVE_SAMPLES', 1))
            max_negative_samples = int(os.environ.get('MAX_NEGATIVE_SAMPLES', 7))
            labels_list = []
            new_batch = []
            for b in batch:
                labels = b.pop('labels', None)
                positive_num = sum(labels)
                negative_num = len(labels) - positive_num
                max_positive = min(positive_num, max_positive_samples)
                max_negative = min(negative_num, max_negative_samples)
                for i in random.sample(range(positive_num), max_positive):
                    new_batch.append({'input_ids': b['input_ids'][i], 'length': b['length'][i]})
                    labels_list.append(1)
                    for j in random.sample(range(negative_num), max_negative):
                        new_batch.append({
                            'input_ids': b['input_ids'][j + positive_num],
                            'length': b['length'][j + positive_num]
                        })
                        labels_list.append(0)

            res = self._data_collator(new_batch, padding_to=padding_to)
            if labels_list:
                res['labels'] = torch.tensor(labels_list, dtype=torch.long)
        else:
            new_batch = []
            for b in batch:
                new_batch.append({'input_ids': b['input_ids']})
            res = self._data_collator(new_batch, padding_to=padding_to)
        return res

    def _seq_cls_data_collator(self,
                               batch: List[Dict[str, Any]],
                               *,
                               padding_to: Optional[int] = None) -> Dict[str, Any]:
        labels = [b.pop('labels') for b in batch if b.get('labels') is not None]
        res = self._data_collator(batch, padding_to=padding_to)
        if labels:
            problem_type = self.config.problem_type
            if problem_type == 'regression':
                labels = torch.tensor(labels, dtype=torch.float32)
            elif problem_type == 'multi_label_classification':
                one_hot_labels = torch.zeros((len(labels), self.config.num_labels), dtype=torch.float32)
                for i, label in enumerate(labels):
                    one_hot_labels[i, label] = 1
                labels = one_hot_labels
            else:
                labels = torch.tensor(labels, dtype=torch.long)
            res['labels'] = labels
        return res

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        assert self.tokenizer.pad_token_id is not None
        padding_side = self.padding_side if self.is_training else 'left'
        padding_right = padding_side == 'right'
        if self.padding_free:
            batch[:] = [self.packing_row(batch)]
            assert 'position_ids' in batch[0], f'batch[0]: {batch[0]}'
        elif self.use_megatron:
            for encoded in batch:
                encoded['position_ids'] = list(range(len(encoded['labels'])))

        res = {}
        if self.padding_free:
            assert len(batch) == 1, f'batch: {batch}'
            for k in ['input_ids', 'labels', 'position_ids', 'loss_scale', 'channel']:
                v = batch[0].get(k)
                if v is not None:
                    res[k] = v if k == 'channel' else [v]
        else:
            inputs_embeds = [b['inputs_embeds'] for b in batch if b.get('inputs_embeds') is not None]
            input_ids = [b['input_ids'] for b in batch if b.get('input_ids') is not None]
            channel = [b.get('channel') for b in batch]

            if inputs_embeds:
                res['inputs_embeds'] = inputs_embeds
            if input_ids:
                res['input_ids'] = input_ids
            if any(channel):
                res['channel'] = channel

            for key in ['labels', 'loss_scale', 'position_ids', 'token_type_ids']:
                val = [b[key] for b in batch if b.get(key) is not None]
                if val:
                    res[key] = val

        keys = [
            'input_ids',
            'inputs_embeds',
            'attention_mask',
            'labels',
            'loss_scale',
            'position_ids',
            'token_type_ids',
            'attention_mask_2d',
        ]
        pad_values = [self.tokenizer.pad_token_id, 0., 0, -100, 0., 0., 0, 0]
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
        if not self.padding_free and seq_lens and ('input_ids' in res or 'inputs_embeds' in res):
            attention_mask_key = 'attention_mask_2d' if self.use_megatron else 'attention_mask'
            res[attention_mask_key] = [torch.ones(seq_len, dtype=torch.int64) for seq_len in seq_lens]
            if self.is_training and self.padding_side == 'left':
                res['position_ids'] = [torch.arange(seq_len, dtype=torch.int64) for seq_len in seq_lens]

        if self.use_megatron:
            # For code simplicity, only the attention_backend 'flash' is supported here.
            if padding_to is not None:
                padding_to = math.ceil(max(seq_lens) / padding_to) * padding_to
            if self.padding_free:
                cp_size = self.sequence_parallel_size
                if cp_size > 1:
                    padding_len = padding_to - seq_lens[0]
                    position_ids = res['position_ids'][0]
                    extended_position_ids = torch.arange(cp_size * 2).repeat(padding_len // (cp_size * 2))
                    if position_ids.ndim == 3:  # compat mrope
                        extended_position_ids = extended_position_ids[None,
                                                                      None, :].expand(position_ids.shape[0], 1, -1)
                    res['position_ids'] = [torch.concat([position_ids, extended_position_ids], dim=-1)]
            else:
                seq_len = max(seq_lens) if padding_to is None else padding_to
                res['attention_mask'] = torch.tril(torch.ones(
                    (len(seq_lens), seq_len, seq_len), dtype=torch.bool)).view(len(seq_lens), 1, seq_len, seq_len)
                assert res['attention_mask'].dtype is torch.bool, f'attention_mask.dtype: {res["attention_mask"].dtype}'
                for i, seq_len in enumerate(seq_lens):
                    res['attention_mask'][i, :, seq_len:] = 0

        for key, pad_value in zip(keys, pad_values):
            if key not in res:
                continue
            if self.use_megatron and not self.padding_free and key == 'attention_mask':
                continue
            if padding_to is not None and not (self.padding_free and key == 'position_ids'
                                               and self.sequence_parallel_size > 1):
                padding_len = padding_to - seq_lens[0]
                if padding_len > 0:
                    res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                        'constant', pad_value)
            if key == 'position_ids' and res[key][0].ndim == 3:
                res[key] = torch.concat(res[key], dim=-1)
            else:
                res[key] = self._pad_sequence(res[key], pad_value)

        # multimodal
        res.update(self._data_collator_mm_data(batch))
        if not self.use_megatron and self.sequence_parallel_size > 1:
            res = self._sp_data_collator(res, padding_to, self.tokenizer, padding_side)

        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # multimodal
        res = {}
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)

        for media_type in ['image', 'video']:
            grid_thw = self.concat_tensor(batch, f'{media_type}_grid_thw', 0)
            if grid_thw is not None:
                res[f'{media_type}_grid_thw'] = grid_thw
        return res

    def _sp_data_collator(self, res, padding_to, tokenizer, padding_side):
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if self.sequence_parallel_size > 1 and input_ids is not None:
            bs, seq_len = input_ids.shape
            if 'position_ids' not in res:
                position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
            else:
                position_ids = res['position_ids']
            assert padding_side == 'right' or bs == 1, 'Sequence parallel only support padding_side=right'
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
            if value is not None:
                res[key] = value
        return res

    def print_inputs(self, inputs: Dict[str, Any]) -> None:
        # Base keys to check
        tokenizer_kwargs = inputs.pop('tokenizer_kwargs', None) or {}
        base_keys = [
            'input', 'labels', 'generate', 'chosen_input', 'chosen_labels', 'rejected_input', 'rejected_labels'
        ]

        # For reranker/embedding modes, also check prefixed keys
        if self.task_type in {'reranker', 'generative_reranker', 'embedding'}:
            prefixes = []
            if self.task_type in {'reranker', 'generative_reranker'}:
                prefixes = ['positive_', 'negative_']
            elif self.task_type == 'embedding':
                prefixes = ['anchor_', 'positive_', 'negative_']

            # Add prefixed keys for reranker/embedding modes
            extended_keys = base_keys.copy()
            for prefix in prefixes:
                for base_key in ['input', 'labels']:
                    extended_keys.append(f'{prefix}{base_key}')

            # Also check for numbered negative keys (negative0_, negative1_, etc.)
            input_keys = list(inputs.keys())
            for key in input_keys:
                if any(key.startswith(f'{prefix}') for prefix in prefixes):
                    # Extract the base key after removing prefix
                    for prefix in prefixes:
                        if key.startswith(prefix):
                            base_key = key[len(prefix):]
                            if base_key in ['input_ids', 'labels'
                                            ] or base_key.rstrip('0123456789_') in ['input', 'labels']:
                                extended_keys.append(key.replace('_ids', ''))
                            break

            keys_to_check = list(set(extended_keys))
        else:
            keys_to_check = base_keys

        for key in keys_to_check:
            # Skip labels completely for certain modes
            if key.endswith('labels') and self.task_type in {'reranker', 'generative_reranker'}:
                continue

            val = inputs.get(key)  # fix val is a tensor
            if val is None:
                val = inputs.get(f'{key}_ids')
            if val is not None:
                key_upper = key.upper()
                logger.info(f'[{key_upper}_IDS] {val}')
                if key.endswith('labels') and self.task_type in {'seq_cls', 'embedding'}:
                    continue
                if isinstance(val, (list, tuple, torch.Tensor)):
                    # Handle nested lists (e.g., for reranker negative samples)
                    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (list, tuple)):
                        val_str = [self.safe_decode(sub_val, **tokenizer_kwargs) for sub_val in val]
                    else:
                        val_str = self.safe_decode(val, **tokenizer_kwargs)
                    logger.info(f'[{key_upper}] {val_str}')
        if inputs.get('loss_scale') is not None:
            val = inputs['loss_scale']
            logger.info(f'[LOSS_SCALE] {val}')

    async def prepare_lmdeploy_pytorch_inputs(self, inputs) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            images[i]['offset'] = len(new_input_ids)
            new_input_ids += [images[i]['image_token_id']] * images[i]['image_tokens']
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['multimodal'] = images

    async def prepare_lmdeploy_turbomind_inputs(self, inputs: Dict[str, Any]) -> None:
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
        inputs['input_embeddings'] = [image.to('cpu') for image in images]
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids

    def _pad_sequence(self, sequences: List[torch.Tensor], padding_value: float = 0.) -> torch.Tensor:
        """Pad sequence by some side

        Args:
            sequences: The input sequences in tensor.
            padding_value: The padding value

        Returns:
            A tensor after padding
        """
        padding_side = self.padding_side if self.is_training and self.task_type != 'generative_reranker' else 'left'
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.shape[0] for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.shape[0]
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def safe_decode(self, input_ids: List[int], **kwargs) -> str:
        if isinstance(self, Template):
            tokenizer = self.tokenizer
            placeholder_tokens = self.placeholder_tokens
        else:
            tokenizer = self
            placeholder_tokens = []

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
                result_str += tokenizer.decode(input_ids[e:s], **kwargs)
            if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
                e = i
                result_str += f'[{input_ids[i - 1]} * {e - s}]'
        if _is_special(input_ids[i]):
            result_str += f'[{input_ids[i]} * {len(input_ids) - s}]'
        else:
            result_str += tokenizer.decode(input_ids[e:], **kwargs)
        return result_str

    @staticmethod
    @contextmanager
    def _patch_flash_attention_forward(modeling_module, position_ids, use_new_func: bool = False):
        _origin_flash_attention_forward = modeling_module._flash_attention_forward

        def _flash_attention_forward(*args, **kwargs):
            if use_new_func:
                from transformers.modeling_flash_attention_utils import (_flash_attention_forward as
                                                                         flash_attention_forward)
                if args and isinstance(args[0], nn.Module):
                    args = args[1:]
                if 'is_causal' not in kwargs:
                    kwargs['is_causal'] = True
            else:
                flash_attention_forward = _origin_flash_attention_forward
            kwargs['position_ids'] = position_ids
            if args and isinstance(args[0], torch.Tensor):
                kwargs['position_ids'] = kwargs['position_ids'].to(args[0].device)
            return flash_attention_forward(*args, **kwargs)

        modeling_module._flash_attention_forward = _flash_attention_forward
        try:
            yield
        finally:
            modeling_module._flash_attention_forward = _origin_flash_attention_forward

    @staticmethod
    def _get_inputs_embeds_hf(inputs_embeds, inputs, visual, processor, config):
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            images = [Image.new('RGB', (32, 32), (0, 0, 0))]
            media_inputs = processor.image_processor(images=images, return_tensors='pt')
            media_inputs = to_device(media_inputs, input_ids.device)
            pixel_values = media_inputs['pixel_values'].type(dtype)
            image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = processor.image_processor.merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    @staticmethod
    def _concat_text_position_ids(position_ids):
        seq_len = position_ids.shape[-1]
        text_position_ids = torch.arange(seq_len, device=position_ids.device).expand(1, *position_ids.shape[1:])
        return torch.concat([text_position_ids, position_ids], dim=0)
