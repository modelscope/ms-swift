# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import inspect
import numpy as np
import re
from PIL import Image
from typing import Any, Dict, List, Literal, Tuple

from ..base import Template
from ..constant import MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context


class Molmo2Template(Template):
    """Native Molmo2 template for image and video understanding."""

    use_model = True

    placeholder_tokens = [
        '<|image|>',
        '<|video|>',
        '<im_patch>',
        '<frame_start>',
        '<frame_end>',
    ]

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|image|>']
        if media_type == 'video':
            return ['<|video|>']
        return []

    @staticmethod
    def _load_video_descriptor(video_item: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if not isinstance(video_item, dict):
            raise TypeError('Molmo2 expects a video descriptor dict produced by the dataset preprocessor.')
        frame_paths = video_item.get('frame_paths') or []
        timestamps = video_item.get('timestamps') or []
        if not frame_paths or not timestamps or len(frame_paths) != len(timestamps):
            raise ValueError('Molmo2 video descriptor requires aligned `frame_paths` and `timestamps`.')

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(np.asarray(image.convert('RGB')))
        frame_array = np.stack(frames, axis=0)
        timestamp_array = np.asarray(timestamps, dtype=np.float32)
        metadata = {
            'frame_paths': frame_paths,
            'source_video': video_item.get('source_video'),
            'num_frames': len(frame_paths),
        }
        return frame_array, timestamp_array, metadata

    @staticmethod
    def _build_messages_for_processor(inputs: StdTemplateInputs) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(inputs.messages)
        image_idx = 0
        video_idx = 0
        for message in messages:
            content = message.get('content', '')
            structured_content: List[Dict[str, Any]] = []
            if not isinstance(content, str):
                message['content'] = content
                continue
            for chunk in re.split(r'(<image>|<video>)', content):
                if not chunk:
                    continue
                if chunk == '<image>':
                    if image_idx >= len(inputs.images):
                        raise ValueError('The number of <image> tags does not match inputs.images.')
                    structured_content.append({'type': 'image', 'image': inputs.images[image_idx]})
                    image_idx += 1
                elif chunk == '<video>':
                    if video_idx >= len(inputs.videos):
                        raise ValueError('The number of <video> tags does not match inputs.videos.')
                    structured_content.append({'type': 'video', 'video': inputs.videos[video_idx]})
                    video_idx += 1
                else:
                    structured_content.append({'type': 'text', 'text': chunk})
            message['content'] = structured_content or [{'type': 'text', 'text': ''}]
        if image_idx != len(inputs.images):
            raise ValueError('Unused images remain after parsing message placeholders.')
        if video_idx != len(inputs.videos):
            raise ValueError('Unused videos remain after parsing message placeholders.')
        return messages

    @staticmethod
    def _load_images(images: List[Any]) -> List[Image.Image]:
        loaded = []
        for image in images:
            if isinstance(image, Image.Image):
                loaded.append(image.convert('RGB'))
            elif isinstance(image, str):
                with Image.open(image) as pil_image:
                    loaded.append(pil_image.convert('RGB'))
            else:
                loaded.append(image)
        return loaded

    @staticmethod
    def _build_video_metadata(frames: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        if len(timestamps) <= 1:
            fps = 1.0
        else:
            median_diff = np.median(np.diff(timestamps))
            fps = 1.0 / float(median_diff) if median_diff > 0 else 1.0
        frames_indices = np.rint(timestamps * fps).astype(int)
        return {
            'total_num_frames': int(frames.shape[0]),
            'fps': float(fps),
            'height': int(frames.shape[1]),
            'width': int(frames.shape[2]),
            'frames_indices': frames_indices.tolist(),
        }

    def _prepare_mm_inputs(self, inputs: StdTemplateInputs) -> Tuple[Dict[str, Any], List[List[int]], List[List[int]]]:
        media_inputs: Dict[str, Any] = {}
        image_expansions: List[List[int]] = []
        video_expansions: List[List[int]] = []
        tokenizer = self.tokenizer

        if inputs.images:
            images = self._load_images(inputs.images)
            image_inputs = self.processor.image_processor(images=images, return_tensors='pt')
            for image_grid, image_num_crops in zip(image_inputs['image_grids'], image_inputs['image_num_crops']):
                image_tokens = self.processor.get_image_tokens(image_grid.cpu().numpy(), int(image_num_crops.item()))
                image_expansions.append(tokenizer.encode(''.join(image_tokens), add_special_tokens=False))
            media_inputs.update(image_inputs)

        if inputs.videos:
            if len(inputs.videos) != 1:
                raise ValueError('Molmo2 currently only supports single-video samples.')
            frames, timestamps, _ = self._load_video_descriptor(inputs.videos[0])
            video_metadata = [self._build_video_metadata(frames, timestamps)]
            video_inputs = self.processor.video_processor(
                videos=[frames],
                video_metadata=video_metadata,
                do_sample_frames=False,
                return_tensors='pt',
                return_metadata=True,
            )
            video_metadata = video_inputs.pop('video_metadata')
            for video_grid, metadata in zip(video_inputs['video_grids'], video_metadata):
                video_string = self.processor.get_video_string(
                    video_grid.cpu().numpy(),
                    np.asarray(metadata.timestamps, dtype=np.float32),
                )
                video_expansions.append(tokenizer.encode(video_string, add_special_tokens=False))
            media_inputs.update(video_inputs)

        return media_inputs, image_expansions, video_expansions

    def _replace_media_placeholders(self, token_ids: List[int], image_expansions: List[List[int]],
                                    video_expansions: List[List[int]]) -> List[int]:
        image_placeholder = self.tokenizer.encode('<|image|>', add_special_tokens=False)
        video_placeholder = self.tokenizer.encode('<|video|>', add_special_tokens=False)
        replaced: List[int] = []
        i = 0
        image_idx = 0
        video_idx = 0
        while i < len(token_ids):
            if video_placeholder and token_ids[i:i + len(video_placeholder)] == video_placeholder:
                if video_idx >= len(video_expansions):
                    raise ValueError('Encountered more <|video|> placeholders than available video expansions.')
                replaced.extend(video_expansions[video_idx])
                video_idx += 1
                i += len(video_placeholder)
                continue
            if image_placeholder and token_ids[i:i + len(image_placeholder)] == image_placeholder:
                if image_idx >= len(image_expansions):
                    raise ValueError('Encountered more <|image|> placeholders than available image expansions.')
                replaced.extend(image_expansions[image_idx])
                image_idx += 1
                i += len(image_placeholder)
                continue
            replaced.append(token_ids[i])
            i += 1
        if image_idx != len(image_expansions):
            raise ValueError('Unused image expansions remain after placeholder replacement.')
        if video_idx != len(video_expansions):
            raise ValueError('Unused video expansions remain after placeholder replacement.')
        return replaced

    def _encode_text_with_media(self, text: str, image_expansions: List[List[int]],
                                video_expansions: List[List[int]]) -> List[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = self._replace_media_placeholders(token_ids, image_expansions, video_expansions)
        attention_mask = np.ones((1, len(token_ids)), dtype=np.int64)
        token_ids_np = np.asarray([token_ids], dtype=np.int64)
        if hasattr(self.processor, 'insert_bos'):
            insert_bos = self.processor.insert_bos
            try:
                parameters = inspect.signature(insert_bos).parameters
            except (TypeError, ValueError):
                parameters = None
            if parameters is not None and len(parameters) >= 4:
                token_ids_np, _ = insert_bos(
                    token_ids_np,
                    attention_mask,
                    self.tokenizer.bos_token_id,
                    self.tokenizer.pad_token_id,
                )
            else:
                token_ids_np, _ = insert_bos(token_ids_np, attention_mask)
        return token_ids_np[0].tolist()

    def _build_token_type_ids(self, input_ids: List[int]) -> List[int]:
        image_token_ids = {int(token_id) for token_id in getattr(self.processor, 'image_token_ids', [])}
        return [1 if token_id in image_token_ids else 0 for token_id in input_ids]

    @staticmethod
    def _extract_text_from_message(message: Dict[str, Any]) -> str:
        content = message.get('content', '')
        if isinstance(content, str):
            return content
        return ''.join(part.get('text', '') for part in content if part.get('type') == 'text')

    def _build_training_prompt_text(self, messages: List[Dict[str, Any]], full_text: str) -> str:
        assistant_text = self._extract_text_from_message(messages[-1])
        if assistant_text and full_text.endswith(assistant_text):
            return full_text[:-len(assistant_text)]
        return self.processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        messages = self._build_messages_for_processor(inputs)
        media_inputs, image_expansions, video_expansions = self._prepare_mm_inputs(inputs)

        if self.is_training and messages and messages[-1]['role'] == 'assistant':
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self._build_training_prompt_text(messages, full_text)
            input_ids = self._encode_text_with_media(full_text, image_expansions, video_expansions)
            prompt_ids = self._encode_text_with_media(prompt_text, image_expansions, video_expansions)
            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError('Molmo2 prompt ids are not a prefix of the full training ids.')
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
            loss_scale = [0.] * len(prompt_ids) + [1.] * (len(input_ids) - len(prompt_ids))
        else:
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self._encode_text_with_media(prompt_text, image_expansions, video_expansions)
            labels = None
            loss_scale = None

        encoded: Dict[str, Any] = {
            'input_ids': input_ids,
            'labels': labels,
            'loss_scale': loss_scale,
            'token_type_ids': self._build_token_type_ids(input_ids),
        }
        encoded.update(media_inputs)
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        for key in ['image_grids', 'video_grids', 'image_token_pooling', 'video_token_pooling', 'image_num_crops']:
            value = self.concat_tensor(batch, key, 0)
            if value is not None:
                res[key] = value
        video_metadata = self.gather_list(batch, 'video_metadata')
        if video_metadata:
            res['video_metadata'] = video_metadata
        return res


register_template(
    TemplateMeta(
        MLLMTemplateType.molmo2,
        prefix=[],
        prompt=['{{QUERY}}'],
        chat_sep=None,
        suffix=[],
        template_cls=Molmo2Template,
    ))
