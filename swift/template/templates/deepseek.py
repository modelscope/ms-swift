# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from PIL import Image, ImageOps
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing import Any, Dict, List, Optional

from swift.utils import get_env_args, get_logger, to_device
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, findall

logger = get_logger()


@dataclass
class DeepseekTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: [['bos_token_id']])
    prompt: Prompt = field(default_factory=lambda: ['User: {{QUERY}}\n\nAssistant:'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: [['eos_token_id']])
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: [['bos_token_id'], '{{SYSTEM}}\n\n'])


register_template(DeepseekTemplateMeta(LLMTemplateType.deepseek, ))

register_template(
    TemplateMeta(
        LLMTemplateType.deepseek_coder,
        prefix=['{{SYSTEM}}'],
        prompt=['### Instruction:\n{{QUERY}}\n### Response:\n'],
        chat_sep=['\n<|EOT|>\n'],
        suffix=['\n<|EOT|>'],
        stop_words=['<|EOT|>'],
        default_system=('You are an AI programming assistant, utilizing the Deepseek Coder model, '
                        'developed by Deepseek Company, and you only answer questions related to computer science. '
                        'For politically sensitive questions, security and privacy issues, '
                        'and other non-computer science questions, you will refuse to answer\n')))


class DeepseekVLTemplate(Template):
    image_placeholder = ['<image_placeholder>']
    skip_prompt = False
    use_model = True
    placeholder_tokens = ['<image_placeholder>']

    image_token_num_per_image: int = 576

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        is_janus = getattr(self, 'is_janus', False)

        encoded = super()._encode(inputs)
        images = inputs.images
        processor = self.processor
        input_ids, labels = encoded['input_ids'], encoded['labels']

        if not inputs.generate_mode:  # understanding task
            idx_list = findall(input_ids, processor.image_id)  # '<image_placeholder>'
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
            encoded = {'output': output, 'input_ids': new_input_ids, 'labels': new_labels}
            return encoded

        else:  # image generation task
            if self.is_training:
                raise NotImplementedError('Only support the inference of generation of Janus series models.')
            sft_format = self.tokenizer.decode(input_ids)
            prompt = sft_format + processor.image_start_tag
            input_ids = processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            encoded = {'input_ids': input_ids, 'labels': labels, 'generate_mode': inputs.generate_mode}
            return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs.get('generate_mode'):
            inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.model_info.torch_dtype)
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            return {'inputs_embeds': inputs_embeds}
        else:
            return inputs

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        gene_img_list = [b.get('generate_mode') for b in batch]
        if all(gene_img_list):
            generate_mode = True
        elif not any(gene_img_list):
            generate_mode = False
        else:
            raise NotImplementedError('Do not support understanding and image generation tasks in one batch.')

        if not generate_mode:
            output = self.fetch_inputs(batch, ['output'])['output']
            batched_output = dict(self.processor.batchify(output))
            res = super()._data_collator(batch, padding_to=padding_to)
            return {**batched_output, **res}
        else:
            res = super()._data_collator(batch, padding_to=padding_to)
            res['generate_mode'] = generate_mode
            return res

    def generate(self, model, *args, **kwargs):
        if not kwargs.get('generate_mode'):
            return super().generate(model, *args, **kwargs)

        else:
            # generate how many number of images for each prompt, it is named parallel_size in the author's code
            parallel_size = kwargs['generation_config'].num_return_sequences
            temperature = kwargs['generation_config'].temperature
            cfg_weight = get_env_args('cfg_weight', float, 5.0)

            input_ids = kwargs['input_ids']  # [bsz, max_input_token_num]
            bsz, max_input_token_num = input_ids.shape
            tokens = torch.zeros((bsz, parallel_size * 2, max_input_token_num),
                                 dtype=torch.int).cuda()  # [bsz, parallel_size*2, max_input_token_num]
            for i in range(parallel_size * 2):
                tokens[:, i, :] = input_ids
                if i % 2 != 0:
                    tokens[:, i, 1:-1] = self.processor.pad_id

            inputs_embeds = model.language_model.get_input_embeddings()(
                tokens)  # [bsz, parallel_size*2, max_input_token_num, 2048]

            generated_tokens = torch.zeros(
                (bsz, parallel_size, self.image_token_num_per_image),
                dtype=torch.int).cuda()  # [bsz, 16, image_token_num_per_image] placeholder for the generated tokens

            # set the first two dimensions into one dimension for batch size
            inputs_embeds = inputs_embeds.reshape(bsz * parallel_size * 2, max_input_token_num, -1)
            generated_tokens = generated_tokens.reshape(bsz * parallel_size, self.image_token_num_per_image)

            for i in range(self.image_token_num_per_image):  # generate the tokens of image in a auto-regression way
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state

                logits = self.model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]

                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)  # [parallel_size, self.image_token_num_per_image]

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = model.prepare_gen_img_embeds(next_token)  # [parallel_size * 2, 2048]
                inputs_embeds = img_embeds.unsqueeze(dim=1)  # [parallel_size * 2, 1, 2048]

            # no need to reset the original first two dimensions, waiting for the update of the upper layer
            # inputs_embeds = inputs_embeds.reshape(bsz, parallel_size*2, -1)
            # generated_tokens = generated_tokens.reshape(bsz, parallel_size, self.image_token_num_per_image)

            return {'sequences': generated_tokens}

    def decode_generate_ids(self, generate_ids: List[int], **kwargs) -> Any:
        if 'template_inputs' not in kwargs or not kwargs['template_inputs'].generate_mode:
            return super().decode_generate_ids(generate_ids, **kwargs)
        else:
            img_size = get_env_args('img_size', int, 384)
            patch_size = 16

            num_to_decode = 1  # for now, generate_ids is a 1D list

            generate_ids = torch.tensor(generate_ids).unsqueeze(0)  # [num_to_decode=1, self.image_token_num_per_image]

            dec = self.model.gen_vision_model.decode_code(
                generate_ids.to(dtype=torch.int),
                shape=[num_to_decode, 8, img_size // patch_size, img_size // patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)  # [num_to_decode, H, W, ch=3]

            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((num_to_decode, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            img_list = []
            for i in range(num_to_decode):
                cur_img = Image.fromarray(visual_img[i])
                img_list.append({'type': 'image', 'image': cur_img})
            return img_list


@dataclass
class DeepseekVLTemplateMeta(DeepseekTemplateMeta):
    default_system: Optional[str] = ('You are a helpful language and vision assistant. '
                                     'You are able to understand the visual content that the user provides, '
                                     'and assist the user with a variety of tasks using natural language.')


register_template(DeepseekVLTemplateMeta(
    MLLMTemplateType.deepseek_vl,
    template_cls=DeepseekVLTemplate,
))


class DeepseekJanus(DeepseekVLTemplate):
    is_janus = True
    image_placeholder = ['<image_placeholder>\n']


register_template(DeepseekVLTemplateMeta(MLLMTemplateType.deepseek_janus, template_cls=DeepseekJanus))


class DeepseekOCR(Template):
    version = 'v1'
    image_placeholder = ['<image>\n']

    def init_env_args(self):
        # Delay loading dynamic modules that require specific transformers versions
        # These will be loaded lazily in _preprocess_image when actually needed
        # This avoids triggering transformers version compatibility issues for vllm backend
        super().init_env_args()
        self._BasicImageTransform = None
        self._dynamic_preprocess = None
        self.crop_mode = get_env_args('crop_mode', bool, True)
        self.base_size = get_env_args('base_size', int, 1024)
        # image_size will be set after detecting version (v1: 640, v2: 768)
        self._image_size_override = get_env_args('image_size', int, None)

    @property
    def image_size(self):
        if self._image_size_override is not None:
            return self._image_size_override
        return 768 if self.version == 'v2' else 640

    @property
    def crop_threshold(self):
        # v1: 640, v2: 768
        return 768 if self.version == 'v2' else 640

    def _load_dynamic_modules(self):
        """Lazily load dynamic modules from model repository."""
        if self._BasicImageTransform is None:
            model_dir = self.model_info.model_dir
            model_type_name = 'deepseekocr2' if self.version == 'v2' else 'deepseekocr'
            self._BasicImageTransform = get_class_from_dynamic_module(f'modeling_{model_type_name}.BasicImageTransform',
                                                                      model_dir)
            self._dynamic_preprocess = get_class_from_dynamic_module(f'modeling_{model_type_name}.dynamic_preprocess',
                                                                     model_dir)

    @property
    def BasicImageTransform(self):
        self._load_dynamic_modules()
        return self._BasicImageTransform

    @property
    def dynamic_preprocess(self):
        self._load_dynamic_modules()
        return self._dynamic_preprocess

    def _preprocess_image(self, images, image_token_id):
        # Code borrowed from
        # https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR/file/view/master/modeling_deepseekocr.py?status=1
        # https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR-2/file/view/master/modeling_deepseekocr2.py?status=1
        crop_mode = self.crop_mode
        patch_size = 16
        downsample_ratio = 4
        valid_img_tokens = 0
        w, h = images[0].size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))
        crop_threshold = self.crop_threshold
        image_size = self.image_size

        image_transform = self.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_list, images_crop_list = [], []
        tokenized_str = []
        images_spatial_crop = []
        for image in images:
            if crop_mode:
                if image.size[0] <= crop_threshold and image.size[1] <= crop_threshold:
                    crop_ratio = [1, 1]
                else:
                    if crop_mode:
                        images_crop_raw, crop_ratio = self.dynamic_preprocess(image)
                    else:
                        crop_ratio = [1, 1]
                """process the global view"""
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size), color=tuple(int(x * 255) for x in image_transform.mean))

                if self.base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif self.base_size == 1280:
                    valid_img_tokens += int(400 * ratio)

                images_list.append(image_transform(global_view).to(torch.bfloat16))
                width_crop_num, height_crop_num = crop_ratio

                images_spatial_crop.append([width_crop_num, height_crop_num])

                if width_crop_num > 1 or height_crop_num > 1:
                    """process the local views"""
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))

                if image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100
                elif image_size == 768:
                    valid_img_tokens += len(images_crop_list) * 144

                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((self.base_size // patch_size) / downsample_ratio)
                """add image tokens"""
                # v1: adds newline token after each row, v2: no newline tokens in rows
                if self.version == 'v2':
                    tokenized_image = ([image_token_id] * num_queries_base) * num_queries_base
                    tokenized_image += [image_token_id]
                    if width_crop_num > 1 or height_crop_num > 1:
                        tokenized_image += ([image_token_id] * (num_queries * width_crop_num)) * (
                            num_queries * height_crop_num)
                else:
                    tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                    tokenized_image += [image_token_id]
                    if width_crop_num > 1 or height_crop_num > 1:
                        tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                            num_queries * height_crop_num)
                tokenized_str.append(tokenized_image)
            else:
                """process the global view"""
                if image_size <= crop_threshold:
                    image = image.resize((image_size, image_size))
                global_view = ImageOps.pad(
                    image, (image_size, image_size), color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                if self.base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif self.base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif self.base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif self.base_size == 512:
                    valid_img_tokens += int(64 * 1)
                elif self.base_size == 768:
                    valid_img_tokens += int(144 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])
                """add image tokens"""
                num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

                # v1: adds newline token after each row, v2: no newline tokens in rows
                if self.version == 'v2':
                    tokenized_image = ([image_token_id] * num_queries) * num_queries
                    tokenized_image += [image_token_id]
                else:
                    tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                    tokenized_image += [image_token_id]
                tokenized_str.append(tokenized_image)
        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        return tokenized_str, images_ori, images_crop, images_spatial_crop

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        image_token = self._tokenize('<image>')
        idx_list = findall(input_ids, image_token)
        if idx_list:
            tokenized_str, images_ori, images_crop, images_spatial_crop = self._preprocess_image(
                inputs.images, image_token[0])
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda i: tokenized_str[i])
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale
            encoded['images'] = [(images_crop, images_ori)]
            encoded['images_seq_mask'] = (torch.tensor(input_ids) == image_token[0])[None]
            encoded['images_spatial_crop'] = images_spatial_crop
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        images = self.gather_list(batch, 'images')
        if images:
            res['images'] = images
        images_seq_mask = [x['images_seq_mask'] for x in batch if x.get('images_seq_mask') is not None]
        images_spatial_crop = self.concat_tensor(batch, 'images_spatial_crop', 0)
        padding_side = self.padding_side if self.is_training else 'left'
        if images_seq_mask:
            max_len = max([x.shape[1] for x in images_seq_mask])
            res['images_seq_mask'] = torch.concat([
                F.pad(x, (0, max_len - x.shape[1]) if padding_side == 'right' else (max_len - x.shape[1], 0))
                for x in images_seq_mask
            ])
        if images_spatial_crop is not None:
            res['images_spatial_crop'] = images_spatial_crop
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.deepseek_ocr,
        prefix=['<｜begin▁of▁sentence｜>'],
        prompt=['{{QUERY}}'],
        chat_sep=None,
        template_cls=DeepseekOCR))


class DeepseekOCR2(DeepseekOCR):
    version = 'v2'


register_template(
    TemplateMeta(
        MLLMTemplateType.deepseek_ocr2,
        prefix=['<｜begin▁of▁sentence｜>'],
        prompt=['{{QUERY}}'],
        chat_sep=None,
        template_cls=DeepseekOCR2))


class UnlimitedOCR(DeepseekOCR):
    image_placeholder = ['<image>']

    def init_env_args(self):
        super().init_env_args()
        self._rswa_window = self.config.sliding_window_size

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        # Official infer_multi uses a single <image> for all images.
        # Expand to N placeholders so DeepseekOCR._encode's 1:1 mapping works.
        n_images = len(inputs.images or [])
        if n_images > 1:
            for msg in inputs.messages:
                content = msg.get('content')
                if isinstance(content, str) and content.count('<image>') == 1:
                    msg['content'] = content.replace('<image>', '<image>' * n_images, 1)
        return super()._encode(inputs)

    # ==================== R-SWA Training Mask (only train) ====================
    @staticmethod
    def _build_rswa_attention_mask(labels, attention_mask_1d, window_size, dtype):
        """Construct an R-SWA mask: Prefix fully visible + Answer sliding window. ⚠️ For training purposes only."""
        batch_size, seq_len = labels.shape
        device = labels.device
        prefix_lens = []
        for i in range(batch_size):
            non_ignored = (labels[i] != -100).nonzero(as_tuple=True)[0]
            prefix_lens.append(non_ignored[0].item() if len(non_ignored) > 0 else seq_len)

        row = torch.arange(seq_len, device=device).view(seq_len, 1)
        col = torch.arange(seq_len, device=device).view(1, seq_len)
        causal = col <= row
        can_attend = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

        for i in range(batch_size):
            p = prefix_lens[i]
            can_attend[i, :p, :] = causal[:p, :]
            can_attend[i, p:, :p] = True
            if p < seq_len:
                answer_len = seq_len - p
                a_row = torch.arange(answer_len, device=device).view(-1, 1)
                a_col = torch.arange(answer_len, device=device).view(1, -1)
                can_attend[i, p:, p:] = causal[p:, p:] & ((a_row - a_col) < window_size)

        valid = attention_mask_1d.bool()
        for i in range(batch_size):
            can_attend[i, :, ~valid[i]] = False
            can_attend[i, ~valid[i], :] = False

        min_val = torch.finfo(dtype).min
        mask = torch.where(can_attend, torch.tensor(0, dtype=dtype, device=device), min_val).to(dtype)
        return mask.unsqueeze(1)

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to=padding_to)
        if not self.is_training or not self._rswa_window or self._rswa_window <= 0:
            return res
        if 'labels' not in res or 'attention_mask' not in res:
            return res

        labels, attn_1d = res['labels'], res['attention_mask']
        if not isinstance(labels, torch.Tensor) or not isinstance(attn_1d, torch.Tensor):
            return res

        res['attention_mask'] = self._build_rswa_attention_mask(labels, attn_1d, self._rswa_window,
                                                                self.model_info.torch_dtype)
        logger.info_once('[UnlimitedOCR] R-SWA windowed mask applied in data_collator')
        return res

    # ==================== Generation Control ====================
    def generate(self, model, *args, **kwargs):
        base_model = self.get_base_model(model)
        config = base_model.config

        _orig_sw = config.sliding_window_size
        config._ring_window = _orig_sw
        config.sliding_window = None

        try:
            ngram_size = get_env_args('no_repeat_ngram_size', int, 0)
            ngram_window = get_env_args('ngram_window', int, 256)
            if ngram_size > 0 and ngram_window > 0:
                ProcessorCls = get_class_from_dynamic_module(
                    'modeling_unlimitedocr.SlidingWindowNoRepeatNgramProcessor', self.model_info.model_dir)
                if ProcessorCls is not None:
                    existing = kwargs.get('logits_processor', []) or []
                    kwargs['logits_processor'] = list(existing) + [ProcessorCls(ngram_size, ngram_window)]

            return super().generate(model, *args, **kwargs)
        finally:
            config.sliding_window = _orig_sw

    # ==================== Post-processing Hooks ====================
    def decode_generate_ids(self, generate_ids: List[int], **kwargs) -> str:
        response = super().decode_generate_ids(generate_ids, **kwargs)
        template_inputs = kwargs.get('template_inputs')
        is_finished = kwargs.get('is_finished', True)

        if is_finished and not self.is_training and template_inputs is not None:
            re_match = get_class_from_dynamic_module('modeling_unlimitedocr.re_match', self.model_info.model_dir)
            if re_match is not None:
                try:
                    matches_ref, matches_images, matches_other = re_match(response)
                    template_inputs._ocr_parsed_refs = {
                        'all': matches_ref,
                        'images': matches_images,
                        'others': matches_other
                    }
                except Exception as e:
                    logger.warning(f'[UnlimitedOCR] Official re_match failed: {e}')
        return response

    def post_process_generate_response(self, response: str, inputs: StdTemplateInputs) -> str:
        if self.is_training:
            return response

        output_dir = (inputs.chat_template_kwargs or {}).get('ocr_output_dir', './ocr_output')
        parsed_refs = getattr(inputs, '_ocr_parsed_refs', None)

        if parsed_refs and inputs.images:
            try:
                import os
                image = inputs.images[0] if isinstance(inputs.images[0], Image.Image) else None
                if image is not None:
                    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

                    draw_fn = get_class_from_dynamic_module('modeling_unlimitedocr.process_image_with_refs',
                                                            self.model_info.model_dir)
                    if draw_fn is not None:
                        result_img = draw_fn(image, parsed_refs['all'], output_dir)
                        result_img.save(os.path.join(output_dir, 'result_with_boxes.jpg'))

                    img_idx = 0
                    for match in parsed_refs['images']:
                        response = response.replace(match, f'![](images/{img_idx}.jpg)\n', 1)
                        img_idx += 1
                    for match in parsed_refs['others']:
                        response = response.replace(match, '')
                    response = response.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
            except Exception as e:
                logger.warning(f'[UnlimitedOCR] Post-process failed: {e}')

        return response.strip()

    def _load_dynamic_modules(self):
        if self._BasicImageTransform is None:
            model_dir = self.model_info.model_dir
            self._BasicImageTransform = get_class_from_dynamic_module('modeling_unlimitedocr.BasicImageTransform',
                                                                      model_dir)
            self._dynamic_preprocess = get_class_from_dynamic_module('modeling_unlimitedocr.dynamic_preprocess',
                                                                     model_dir)


register_template(
    TemplateMeta(
        MLLMTemplateType.unlimited_ocr,
        prefix=[['bos_token_id']],
        prompt=['{{QUERY}}'],
        chat_sep=None,
        template_cls=UnlimitedOCR,
    ))


@dataclass
class DeepseekV2_5TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<｜begin▁of▁sentence｜>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<｜User｜>{{QUERY}}<｜Assistant｜>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])
    suffix: Prompt = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])


register_template(DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_v2_5))

register_template(DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_r1, is_thinking=True, thinking_prefix='<think>\n'))


class DeepseekV3_1Template(Template):
    jinja_enable_thinking_key = 'thinking'
    non_thinking_prefix_only_after_user = True


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v3_1,
        agent_template='deepseek_v3_1',
        is_thinking=True,
        template_cls=DeepseekV3_1Template,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))

# Reasoning-effort prefixes, prepended at the very beginning of the conversation
# (before the system content) when thinking is enabled. Naming follows the prompt text
# rather than the level, because the level each one maps to differs between releases:
# `ABSOLUTE_MAX` is `max` for V4-Flash/V4-Pro (preview) but `high` for V4-Flash-0731.
REASONING_EFFORT_ABSOLUTE_MAX = (
    'Reasoning Effort: Absolute maximum with no shortcuts permitted.\n'
    'You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve '
    'the root cause, rigorously stress-testing your logic against all potential paths, edge cases, '
    'and adversarial scenarios.\n'
    'Explicitly write out your entire deliberation process, documenting every intermediate step, '
    'considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n')

REASONING_EFFORT_BEYOND_MAX = (
    'Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n'
    'You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: '
    'exhaustively decompose the problem into its most fundamental components, trace every causal chain '
    'to its root, and resolve the underlying cause rather than any surface symptom.\n'
    'Do not stop reasoning until you have independently verified the solution from multiple angles and '
    'are certain that no assumption remains unchecked and no error remains undiscovered.\n\n')


class DeepseekV4Template(DeepseekV3_1Template):
    # V4-Flash / V4-Pro (preview) ship two thinking levels; `high` adds no prefix.
    reasoning_effort_prompts = {'high': '', 'max': REASONING_EFFORT_ABSOLUTE_MAX}
    default_reasoning_effort = 'high'

    def init_env_args(self):
        super().init_env_args()
        self.reasoning_effort = self._check_reasoning_effort(get_env_args('reasoning_effort', str, None))
        if self.reasoning_effort is None:
            self.reasoning_effort = self.default_reasoning_effort if self.enable_thinking else None
        self.enable_thinking = self.reasoning_effort in self.reasoning_effort_prompts
        self.chat_template_kwargs['reasoning_effort'] = self.reasoning_effort

    def _check_reasoning_effort(self, reasoning_effort):
        """Drop an unknown level so that it falls back to the default instead of disabling thinking.

        Every accepted level is a thinking level, so an unrecognized value would otherwise be
        indistinguishable from "thinking off" and silently turn reasoning off.
        """
        if reasoning_effort is not None and reasoning_effort not in self.reasoning_effort_prompts:
            logger.warning(f'Ignoring unknown reasoning_effort: {reasoning_effort!r}. '
                           f'Expected one of {list(self.reasoning_effort_prompts)}.')
            return None
        return reasoning_effort

    def _get_reasoning_effort(self, inputs=None):
        reasoning_effort = None if inputs is None else inputs.chat_template_kwargs.get('reasoning_effort')
        reasoning_effort = self._check_reasoning_effort(reasoning_effort)
        if reasoning_effort is None:
            reasoning_effort = self.reasoning_effort
        return reasoning_effort

    def _get_enable_thinking(self, inputs=None):
        reasoning_effort = None if inputs is None else inputs.chat_template_kwargs.get('reasoning_effort')
        reasoning_effort = self._check_reasoning_effort(reasoning_effort)
        if reasoning_effort is not None:
            return reasoning_effort in self.reasoning_effort_prompts
        return super()._get_enable_thinking(inputs)

    def _get_system(self, inputs):
        system = super()._get_system(inputs)
        if self._get_enable_thinking(inputs):
            prefix = self.reasoning_effort_prompts.get(self._get_reasoning_effort(inputs)) or ''
            if prefix:
                system = prefix + (system or '')
        return system

    def _remove_history_thinking(self, inputs) -> None:
        # The official encoding disables `drop_thinking` once tools are defined: tool-calling
        # conversations keep the reasoning of every turn so the model can track multi-step
        # reasoning across tool calls.
        if inputs.tools:
            return
        super()._remove_history_thinking(inputs)


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v4,
        agent_template='deepseek_v4',
        is_thinking=True,
        template_cls=DeepseekV4Template,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))


class DeepseekV4FlashTemplate(DeepseekV4Template):
    # V4-Flash-0731 ships three thinking levels and shifts the prefixes one level down:
    # what `max` meant in the preview release is `high` here, and `max` gets a stronger text.
    # `low` is the default and adds no prefix (it is still a thinking level).
    reasoning_effort_prompts = {
        'low': '',
        'high': REASONING_EFFORT_ABSOLUTE_MAX,
        'max': REASONING_EFFORT_BEYOND_MAX,
    }
    default_reasoning_effort = 'low'


register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v4_flash,
        agent_template='deepseek_v4',
        is_thinking=True,
        template_cls=DeepseekV4FlashTemplate,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))


class DeepseekVL2Template(DeepseekVLTemplate):
    image_placeholder = ['<image>\n']
    placeholder_tokens = ['<image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from deepseek_vl2.models.processing_deepseek_vl_v2 import VLChatProcessorOutput
        encoded = Template._encode(self, inputs)
        images = inputs.images
        processor = self.processor
        input_ids, labels = encoded['input_ids'], encoded['labels']
        images_seq_mask = [False] * len(input_ids)
        idx_list = findall(input_ids, processor.image_token_id)  # '<image>'
        _, images_list, _, images_spatial_crop, num_image_tokens = processor.tokenize_with_images(
            '<image>' * len(images), images, cropping=len(images) <= 2)
        new_num_tokens = 0
        for idx, n_image_tokens in zip(idx_list, num_image_tokens):
            image_tokens = [processor.image_token_id] * n_image_tokens
            input_ids = input_ids[:idx] + image_tokens + input_ids[idx + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * n_image_tokens + labels[idx + 1:]
            images_seq_mask = images_seq_mask[:idx] + [True] * n_image_tokens + images_seq_mask[idx + 1:]
            new_num_tokens += n_image_tokens - 1

        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=torch.tensor(input_ids),
            target_ids=torch.tensor(input_ids),
            images=torch.stack(images_list) if images_list else torch.zeros((0, 3, 384, 384)),
            images_seq_mask=torch.tensor(images_seq_mask),
            images_spatial_crop=torch.tensor(images_spatial_crop),
            num_image_tokens=num_image_tokens)
        output.images = output.images.to(dtype=self.model_info.torch_dtype)
        encoded = {'output': output, 'input_ids': input_ids, 'labels': labels}
        return encoded

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs['images_seq_mask'] = inputs['images_seq_mask'].to(torch.bool)
        inputs['images_spatial_crop'] = inputs['images_spatial_crop'].to(torch.long)
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        return {'inputs_embeds': inputs_embeds}


register_template(
    DeepseekV2_5TemplateMeta(
        MLLMTemplateType.deepseek_vl2,
        prompt=['<|User|>: {{QUERY}}\n\n<|Assistant|>:'],
        template_cls=DeepseekVL2Template,
    ))

register_template(
    DeepseekVLTemplateMeta(
        MLLMTemplateType.deepseek_janus_pro,
        prompt=['<|User|>: {{QUERY}}\n\n<|Assistant|>:'],
        template_cls=DeepseekJanus))


class DeepseekV4VisionTemplate(DeepseekV4FlashTemplate):
    image_placeholder = ['<｜deepseek_image｜>']
    placeholder_tokens = ['<｜deepseek_image｜>']
    skip_prompt = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vit = None
        self._aligner = None
        self._image_params = None
        self._vit_args = None
        self._shard_map = None
        self._shard_map_dir = None
        import threading
        self._mm_lock = threading.Lock()

    def _resolve_model_dir(self):
        """Return the model_dir that contains inference/ and encoding/ subdirs.

        If the checkpoint dir lacks them, fall back to the original model dir
        stored in args.json (``model_dir`` key).
        """
        import json
        import os
        model_dir = self.model_info.model_dir
        if os.path.isdir(os.path.join(model_dir, 'inference')):
            return model_dir
        # Try args.json from the checkpoint
        args_path = os.path.join(model_dir, 'args.json')
        if os.path.exists(args_path):
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            orig_dir = old_args.get('model_dir') or old_args.get('model')
            if orig_dir and os.path.isdir(os.path.join(orig_dir, 'inference')):
                return orig_dir
        return model_dir

    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        old_kwargs = to_device(kwargs, model.device)
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        for k, v in old_kwargs.items():
            if k in {
                    'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                    'max_length_q', 'max_length_k', 'cu_seq_lens_q', 'cu_seq_lens_k', 'mm_token_type_ids'
            } and k not in kwargs:
                kwargs[k] = v
        # NOTE: do NOT pop input_ids — DeepSeek-V4 hash-MoE needs it.

        if 'inputs_embeds' in kwargs and 'input_ids' in kwargs:
            base_model = self.get_base_model(model)
            inner = getattr(base_model, 'model', base_model)
            if not getattr(inner, '_swift_dsv4_mm_patched', False):
                _orig_forward = inner.forward

                def _mm_forward(self, *a, **kw):
                    ids = kw.get('input_ids')
                    embeds = kw.get('inputs_embeds')
                    if ids is not None and embeds is not None:
                        kw['input_ids'] = None
                        _mm_ids = ids
                        _orig_layers = self.layers

                        class _MMLayerWrapper(torch.nn.Module):

                            def __init__(self, real_layer):
                                super().__init__()
                                self._real = real_layer

                            def forward(self, *la, **lkw):
                                if 'input_ids' not in lkw or lkw['input_ids'] is None:
                                    lkw['input_ids'] = _mm_ids
                                return self._real(*la, **lkw)

                        with self._swift_template._mm_lock:
                            self.layers = torch.nn.ModuleList([_MMLayerWrapper(layer) for layer in _orig_layers])
                            try:
                                return _orig_forward(*a, **kw)
                            finally:
                                self.layers = _orig_layers

                    return _orig_forward(*a, **kw)

                import types
                inner.forward = types.MethodType(_mm_forward, inner)
                inner._swift_dsv4_mm_patched = True
                inner._swift_template = self

        base_model = self.get_base_model(model)
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    def _load_vision_modules(self):
        import importlib
        import os
        model_dir = self._resolve_model_dir()
        inference_dir = os.path.join(model_dir, 'inference')
        encoding_dir = os.path.join(model_dir, 'encoding')
        for d in [inference_dir, encoding_dir]:
            if d not in sys.path:
                sys.path.insert(0, d)
        return importlib.import_module('image_processor')

    def _get_vision_args(self):
        from types import SimpleNamespace
        config = self.config
        return SimpleNamespace(
            vocab_size=config.vocab_size,
            vision_patch_size=config.vision_patch_size,
            vision_dim=config.vision_dim,
            vision_n_heads=config.vision_n_heads,
            vision_inter_dim=config.vision_inter_dim,
            vision_n_layers=config.vision_n_layers,
            vision_rope_theta=config.vision_rope_theta,
            vision_downsample_ratio=config.vision_downsample_ratio,
            vision_max_n_token=config.vision_max_n_token,
            vision_min_pixels=config.vision_min_pixels,
            vision_max_wh_ratio=config.vision_max_wh_ratio,
            dim=config.hidden_size,
        )

    def _process_images(self, images, input_ids, labels, loss_scale):
        image_processor = self._load_vision_modules()
        args = self._get_vision_args()

        # Token id of the placeholder
        placeholder_id = self.tokenizer.convert_tokens_to_ids('<｜deepseek_image｜>')
        idx_list = findall(input_ids, placeholder_id)

        new_input_ids, new_labels, new_loss_scale = [], [], []
        image_inputs = []
        lo = 0

        img_iter = iter(images)
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            if loss_scale is not None:
                new_loss_scale += loss_scale[lo:hi]

            # Load and process the image
            img_obj = next(img_iter)
            if isinstance(img_obj, str):
                record = {'url': img_obj}
            elif isinstance(img_obj, Image.Image):
                import base64
                import io
                buf = io.BytesIO()
                img_obj.save(buf, format='PNG')
                record = {'data': base64.b64encode(buf.getvalue()).decode()}
            else:
                record = img_obj

            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = image_processor.load_image(record, args)
            types, perm = image_processor.build_image_block(n_llm_h, n_llm_w, len(new_input_ids))

            # Sentinel tokens: vocab_size + type
            sentinel_tokens = (args.vocab_size + types).tolist()
            new_input_ids += sentinel_tokens
            if labels is not None:
                new_labels += [-100] * len(sentinel_tokens)
            if loss_scale is not None:
                new_loss_scale += [0.] * len(sentinel_tokens)

            # ImageInput: store start, patches, n_vit_h, n_vit_w, types, perm
            image_inputs.append(
                image_processor.ImageInput(
                    start=len(new_input_ids) - len(sentinel_tokens),
                    patches=patches,
                    n_vit_h=n_vit_h,
                    n_vit_w=n_vit_w,
                    types=types,
                    perm=perm,
                ))

            lo = hi + 1

        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        if loss_scale is not None:
            new_loss_scale += loss_scale[lo:]

        return new_input_ids, new_labels, new_loss_scale, image_inputs

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        images = inputs.images
        if not images:
            return encoded

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale')

        new_input_ids, new_labels, new_loss_scale, image_inputs = self._process_images(
            images, input_ids, labels, loss_scale)

        encoded['input_ids'] = new_input_ids
        encoded['labels'] = new_labels
        if loss_scale is not None:
            encoded['loss_scale'] = new_loss_scale
        encoded['image_inputs'] = image_inputs
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        image_inputs = [b['image_inputs'] for b in batch if b.get('image_inputs') is not None]
        if image_inputs:
            res['image_inputs'] = image_inputs
        return res

    @staticmethod
    def _get_embed_fn(base_model):
        """Recursively find the embedding function on the model."""
        for attr in ['embed', 'embed_tokens']:
            obj = base_model
            for _ in range(4):  # search up to 4 levels deep
                if hasattr(obj, attr):
                    fn = getattr(obj, attr)
                    if callable(fn):
                        return fn
                obj = getattr(obj, 'model', None)
                if obj is None:
                    break
        raise AttributeError('Cannot find embed/embed_tokens on model or any nested .model attribute')

    @staticmethod
    def _get_merge_fn(base_model):
        """Find merge_image_embeddings on the model or a nested .model attribute."""
        obj = base_model
        for _ in range(4):
            if hasattr(obj, 'merge_image_embeddings'):
                return obj.merge_image_embeddings
            obj = getattr(obj, 'model', None)
            if obj is None:
                break
        return None

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_inputs = inputs.get('image_inputs')
        if not image_inputs:
            return inputs

        base_model = self.get_base_model(model)
        input_ids = inputs['input_ids']
        config = getattr(base_model, 'config', None) or getattr(base_model.model, 'config', None)
        vocab_size = config.vocab_size
        masked_ids = torch.masked_fill(input_ids, input_ids >= vocab_size, 0)
        embed_fn = self._get_embed_fn(base_model)
        h = embed_fn(masked_ids)

        merge_fn = self._get_merge_fn(base_model)
        if merge_fn is not None:
            merge_fn(image_inputs, h)
        else:
            self._merge_image_embeds_custom(h, input_ids, image_inputs, base_model)

        return {'inputs_embeds': h, 'input_ids': masked_ids}

    def _ensure_vision_modules(self, h, config):
        import json
        import os
        from safetensors.torch import safe_open as _safe_open
        from types import SimpleNamespace

        if self._vit is not None and self._aligner is not None:
            return

        dtype = h.dtype
        device = h.device

        # Resolve which model_dir has inference/ and encoding/
        model_dir = self._resolve_model_dir()
        inference_dir = os.path.join(model_dir, 'inference')
        encoding_dir = os.path.join(model_dir, 'encoding')
        for d in [inference_dir, encoding_dir]:
            if d not in sys.path:
                sys.path.insert(0, d)

        import image_processor  # noqa: F401  (ensures module is importable)
        import vision as vision_mod

        args = SimpleNamespace(
            vision_patch_size=config.vision_patch_size,
            vision_dim=config.vision_dim,
            vision_n_heads=config.vision_n_heads,
            vision_inter_dim=config.vision_inter_dim,
            vision_n_layers=config.vision_n_layers,
            vision_rope_theta=config.vision_rope_theta,
            vision_downsample_ratio=config.vision_downsample_ratio,
            dim=config.hidden_size,
        )
        self._vit_args = args

        vit = vision_mod.ViT(args).to(device=device, dtype=dtype)
        aligner = vision_mod.Aligner(args).to(device=device, dtype=dtype)

        _orig_gcs = vision_mod.get_vision_cos_sin

        def _gcs(n_h, n_w, dim, theta):
            cos, sin = _orig_gcs(n_h, n_w, dim, theta)
            return cos.to(device), sin.to(device)

        vision_mod.get_vision_cos_sin = _gcs

        weight_dir = self.model_info.model_dir
        index_path = os.path.join(weight_dir, 'model.safetensors.index.json')
        shard_map = {}
        if os.path.exists(index_path):
            with open(index_path) as _f:
                shard_map = json.load(_f).get('weight_map', {})

        def _load_ckpt_tensor(key):
            """Load a single tensor from the checkpoint, trying several key forms."""
            for ckpt_key in [key, f'model.{key}']:
                shard = shard_map.get(ckpt_key)
                if shard is None:
                    continue
                with _safe_open(os.path.join(weight_dir, shard), framework='pt') as _sf:
                    return _sf.get_tensor(ckpt_key).to(dtype=dtype, device=device)
            # Fallback: scan shards when no index exists.
            for _fn in sorted(os.listdir(weight_dir)):
                if not _fn.endswith('.safetensors'):
                    continue
                try:
                    with _safe_open(os.path.join(weight_dir, _fn), framework='pt') as _sf:
                        for ckpt_key in [key, f'model.{key}']:
                            if ckpt_key in _sf.keys():
                                return _sf.get_tensor(ckpt_key).to(dtype=dtype, device=device)
                except Exception:
                    continue
            return None

        # Load ViT weights (checkpoint keys: vision.*)
        vit_sd = {}
        for name, _ in vit.state_dict().items():
            tensor = _load_ckpt_tensor(f'vision.{name}')
            if tensor is not None:
                vit_sd[name] = tensor
        if vit_sd:
            vit.load_state_dict(vit_sd, strict=False)

        # Load Aligner weights (checkpoint keys: aligner.*)
        aligner_sd = {}
        for name, _ in aligner.state_dict().items():
            tensor = _load_ckpt_tensor(f'aligner.{name}')
            if tensor is not None:
                aligner_sd[name] = tensor
        if aligner_sd:
            aligner.load_state_dict(aligner_sd, strict=False)

        # Load image special-token embeddings (checkpoint keys: image_start etc.)
        for key in ['image_start', 'image_end', 'image_pad', 'image_newline']:
            tensor = _load_ckpt_tensor(key)
            if tensor is not None:
                setattr(vit, key, nn.Parameter(tensor))

        vit.eval()
        aligner.eval()
        self._vit = vit
        self._aligner = aligner

        IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
        self._image_params = torch.stack([
            vit.image_start,
            vit.image_pad,
            vit.image_pad,  # placeholder for IMAGE; overwritten by aligner output at merge time
            vit.image_newline,
            vit.image_end,
        ]).to(
            device=device, dtype=dtype)

    def _merge_image_embeds_custom(self, h, input_ids, image_inputs, base_model):
        config = base_model.config if hasattr(base_model, 'config') else base_model.model.config

        # Lazily initialise and cache ViT / Aligner / params
        self._ensure_vision_modules(h, config)

        vit = self._vit
        aligner = self._aligner
        params = self._image_params

        IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
        vocab_size = config.vocab_size
        dtype = h.dtype
        device = h.device

        all_images = []
        for sample in image_inputs:
            if sample is not None:
                all_images.extend(sample)

        img_idx = 0
        for i in range(input_ids.shape[0]):
            row_ids = input_ids[i]
            start_mask = row_ids == vocab_size + IMAGE_START
            start_positions = start_mask.nonzero(as_tuple=False).squeeze(-1)

            for start_pos in start_positions:
                if img_idx >= len(all_images):
                    break
                img = all_images[img_idx]
                img_idx += 1

                sp = start_pos.item()
                ep = sp
                row_len = row_ids.shape[0]
                while ep < row_len and row_ids[ep].item() >= vocab_size:
                    ep += 1
                if ep == sp:
                    continue

                types = (row_ids[sp:ep] - vocab_size).to(torch.int64)
                embeds = aligner(
                    vit(img.patches.to(device=device, dtype=dtype), img.n_vit_h, img.n_vit_w),
                    img.n_vit_h,
                    img.n_vit_w,
                )[img.perm.to(device)].to(dtype)
                block = params[types].clone()
                block[types == IMAGE] = embeds
                h[i, sp:ep] = block


register_template(
    DeepseekV2_5TemplateMeta(
        MLLMTemplateType.deepseek_v4_flash_vision,
        agent_template='deepseek_v4',
        is_thinking=True,
        template_cls=DeepseekV4VisionTemplate,
        thinking_prefix='<think>',
        non_thinking_prefix='</think>',
        history_thinking_prefix='</think>'))
