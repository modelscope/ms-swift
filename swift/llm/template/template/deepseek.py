# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.utils import get_env_args
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Prompt, findall
from .utils import ThinkingTemplate


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

    def decode(self, generate_ids: List[int], **kwargs) -> Any:
        if 'template_inputs' not in kwargs or not kwargs['template_inputs'].generate_mode:
            return super().decode(generate_ids, **kwargs)
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
    image_placeholder = ['<image>\n']

    def init_env_args(self):
        model_dir = self.model_info.model_dir
        self.BasicImageTransform = get_class_from_dynamic_module('modeling_deepseekocr.BasicImageTransform', model_dir)
        self.dynamic_preprocess = get_class_from_dynamic_module('modeling_deepseekocr.dynamic_preprocess', model_dir)
        self.crop_mode = get_env_args('crop_mode', bool, True)
        self.base_size = get_env_args('base_size', int, 1024)
        self.image_size = get_env_args('image_size', int, 640)

    def _preprocess_image(self, images, image_token_id):
        # Code borrowed from
        # https://modelscope.cn/models/deepseek-ai/DeepSeek-OCR/file/view/master/modeling_deepseekocr.py?status=1
        crop_mode = self.crop_mode
        patch_size = 16
        downsample_ratio = 4
        valid_img_tokens = 0
        w, h = images[0].size
        ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

        image_transform = self.BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        images_list, images_crop_list = [], []
        tokenized_str = []
        images_spatial_crop = []
        for image in images:
            if crop_mode:
                if image.size[0] <= 640 and image.size[1] <= 640:
                    crop_ratio = [1, 1]
                else:
                    if crop_mode:
                        images_crop_raw, crop_ratio = self.dynamic_preprocess(image)
                    else:
                        crop_ratio = [1, 1]
                """process the global view"""
                # image = image.resize((base_size, base_size))
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

                if self.image_size == 640:
                    valid_img_tokens += len(images_crop_list) * 100

                num_queries = math.ceil((self.image_size // patch_size) / downsample_ratio)
                num_queries_base = math.ceil((self.base_size // patch_size) / downsample_ratio)
                """add image tokens"""

                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                tokenized_image += [image_token_id]
                if width_crop_num > 1 or height_crop_num > 1:
                    tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                        num_queries * height_crop_num)
                tokenized_str.append(tokenized_image)
            else:
                """process the global view"""
                if self.image_size <= 640:
                    image = image.resize((self.image_size, self.image_size))
                # else:
                global_view = ImageOps.pad(
                    image, (self.image_size, self.image_size), color=tuple(int(x * 255) for x in image_transform.mean))
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                if self.base_size == 1024:
                    valid_img_tokens += int(256 * ratio)
                elif self.base_size == 1280:
                    valid_img_tokens += int(400 * ratio)
                elif self.base_size == 640:
                    valid_img_tokens += int(100 * 1)
                elif self.base_size == 512:
                    valid_img_tokens += int(64 * 1)

                width_crop_num, height_crop_num = 1, 1

                images_spatial_crop.append([width_crop_num, height_crop_num])
                """add image tokens"""
                num_queries = math.ceil((self.image_size // patch_size) / downsample_ratio)

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


@dataclass
class DeepseekV2_5TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<｜begin▁of▁sentence｜>{{SYSTEM}}'])
    prompt: Prompt = field(default_factory=lambda: ['<｜User｜>{{QUERY}}<｜Assistant｜>'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])
    suffix: Prompt = field(default_factory=lambda: ['<｜end▁of▁sentence｜>'])


register_template(DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_v2_5))


class DeepseekV3_1Template(ThinkingTemplate):
    no_think_prefix = '</think>'
    history_think_prefix = '</think>'
    add_no_think_prefix_after_tool = False


register_template(
    DeepseekV2_5TemplateMeta(LLMTemplateType.deepseek_r1, template_cls=ThinkingTemplate, response_prefix='<think>\n'))

# enable thinking: response_prefix='<think>'
register_template(
    DeepseekV2_5TemplateMeta(
        LLMTemplateType.deepseek_v3_1,
        template_cls=DeepseekV3_1Template,
        response_prefix='</think>',
        agent_template='deepseek_v3_1'))


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
