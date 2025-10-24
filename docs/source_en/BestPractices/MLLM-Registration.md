
# Best Practices for Registering Multimodal Models

This document introduces how to register a multimodal model in ms-swift and successfully perform inference and training. Using Qwen2.5-Omni as an example, we will register a new model_type and template `my_qwen2_5_omni`, supporting training with text, images, videos, and audio. Since Qwen2.5-Omni is already registered in ms-swift, we can use our custom components by explicitly specifying the model_type and template.

## Environment Setup

```shell
# Avoid future incompatibilities with documentation
pip install "ms-swift>=3.9,<3.10"

pip install "transformers==4.57.*" "qwen_omni_utils==0.0.8"
```

## Model Registration

First, we need to register the model to obtain the model and processor.

```python
from swift.llm import (
    register_model, ModelMeta, ModelGroup, Model, register_model_arch, MultiModelKeys,
    get_model_tokenizer_with_flash_attn, get_model_tokenizer
)
from swift.llm.model.model.qwen import patch_qwen_vl_utils
from swift.llm.model.utils import use_submodel_func
from swift.llm.model.patcher import patch_get_input_embeddings
from swift.utils import get_env_args


register_model_arch(
    MultiModelKeys(
        'my_qwen2_5_omni',
        # `freeze_llm`, `freeze_vit`, `freeze_aligner` behavior is determined by the values below.
        # For example: full parameter training, if `freeze_vit=True`, it will freeze parameters of model layers prefixed with `thinker.audio_tower` and `thinker.visual`.
        # LoRA training, if `freeze_vit=False`, it will additionally add LoRA to Linear layers prefixed with `thinker.audio_tower` and `thinker.visual`.
        language_model='thinker.model',
        vision_tower=['thinker.audio_tower', 'thinker.visual'],
        aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        # Generator parts will never be trained or remain frozen.
        generator=['talker', 'token2wav'],
    ))

def get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
    from qwen_omni_utils import vision_process
    print('Run my_qwen2_5_omni...')
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5OmniForConditionalGeneration
    # Customize how to get tokenizer and config in `get_model_tokenizer_with_flash_attn`
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen2_5OmniConfig.from_pretrained(model_dir, trust_remote_code=True)
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    # Control constants in qwen_omni_utils library via environment variables, e.g., `MAX_PIXELS`, etc.
    patch_qwen_vl_utils(vision_process)
    # Recommended: Use this function to get model and tokenizer. Avoid using AutoModelForCausalLM directly (may cause incompatibility).
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        # For multimodal model consistency, we replace the model's forward/generate functions with those of its language_model.
        # Handle additional parts separately.
        use_submodel_func(model, 'thinker')
        # Some custom settings for model/config (usually not needed; configure based on specific model if errors occur during training/inference)
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        model.config.talker_config.pad_token_id = None
        # Avoid inplace operations on leaf_variable during training (replacing parts of input_embeds with images_embeds)
        patch_get_input_embeddings(model.thinker.visual, 'patch_embed')
    # Must return model and processor (multimodal) / tokenizer (text-only)
    return model, processor


register_model(
    ModelMeta(
        'my_qwen2_5_omni',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        'my_qwen2_5_omni',
        # Function to get model and processor.
        get_model_tokenizer_qwen2_5_omni,
        is_multimodal=True,  # Whether it's a multimodal model
        model_arch='my_qwen2_5_omni',  # Usually set only for multimodal models
        # Used for automatic model_type matching
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration'],
        # Used to prompt users about dependency versions (can be removed)
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        # Used to prompt users (can be removed)
        tags=['vision', 'video', 'audio'],
        # Additional files to save during full parameter training/merge-lora
        additional_saved_files=['spk_dict.pt'],
    ))

if __name__ == '__main__':
    # Test and debug
    model, processor = get_model_tokenizer('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni')
```

## Template Registration

Second, we need to register a template to customize how text, images, videos, and audio are preprocessed (`_encode` and `_data_collator` methods). This is a key module for ms-swift's support of multimodal model training. Preprocessing methods should reference transformers inference implementation and align with it.

Template functions:

1. Support normal inference and training, preprocess text and multimodal information, and support grounding tasks.
2. Support padding_free and packing training.
3. Support mixed modality data training.

```python
from swift.llm import (
    register_template, Template, get_packed_seq_params, to_float_dtype, TemplateMeta,
    get_template, get_model_tokenizer
)
from transformers.integrations import is_deepspeed_zero3_enabled
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall
from swift.llm.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled
from functools import partial
from typing import Dict, List, Any, Literal, Optional
import torch

logger = get_logger()

class Qwen2_5OmniTemplate(Template):
    use_model = True  # Whether model participation is required during preprocessing
    # Note: Not all multimodal models support padding_free/packing. Models in `transformers` library usually support it
    support_padding_free = True  # Whether padding_free and packing are supported (multimodal models)
    norm_bbox = 'none'  # Whether grounding tasks use absolute or norm1000 coordinates

    # These tokens will not be truncated (e.g., when setting `--truncation_strategy left/right`)
    # and will be printed in abbreviated form (calling `template.safe_decode`)
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor) -> None:
        """Initialize some required constants when initializing the processor"""
        if processor is None:
            return
        super().init_processor(processor)
        from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
        default = Qwen2_5OmniProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)
        # See grounding dataset customization documentation for `QWENVL_BBOX_FORMAT` meaning
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')


    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """Load multimodal data and replace generic multimodal tags.
        For example: image tag from `<image>` -> `<|vision_bos|><|IMAGE|><|vision_eos|>`"""
        # Loading multimodal data can also be done in the `_encode` function, whichever is more convenient.
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':  # No processing needed in 'vllm' inference scenario
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            _video = fetch_video({'video': video})
            if isinstance(_video, torch.Tensor):
                _video = _video.to(torch.uint8)
            inputs.videos[index] = _video
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            else:
                return ['<|vision_bos|><|VIDEO|><|vision_eos|>']


    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace generic tag for grounding tasks: `<ref-object>`"""
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace generic tag for grounding tasks: `<bbox>`"""
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Support packing & mrope.

        Usually no need to inherit this function; here for customizing mrope's position_ids."""
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_new_tokens_use_audio_in_video(self, i, *, video_grid_thw, video_second_per_grid, audio_lengths,
                                           video_token_id, audio_token_id):
        """Helper function to support `use_audio_in_video` being True"""
        merge_size = self.processor.image_processor.merge_size
        grid_thw = video_grid_thw[i]
        height = grid_thw[1] // merge_size
        width = grid_thw[2] // merge_size
        audio_token_indices = torch.arange(audio_lengths[i])
        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)

        video_token_indices = torch.broadcast_to(video_token_indices,
                                                 (video_token_indices.shape[0], height, width)).reshape(-1)
        video_token_indices = (video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
        video_chunk_indexes = self.processor.get_chunked_index(video_token_indices, tokens_per_chunk)
        audio_chunk_indexes = self.processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

        res = []
        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
            if j < len(video_chunk_indexes):
                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                res += video_token_id * video_seq_length
            if j < len(audio_chunk_indexes):
                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                res += audio_token_id * audio_seq_length
        return res


    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """This determines how to convert text/images/audios/videos -> input_ids, labels, loss_scale, and multimodal content like pixel_values
        Processing logic can usually be borrowed from the corresponding model's preprocessing code implementation.
        Recommended: Perform inference alignment first, then training"""
        encoded = Template._encode(self, inputs)  # Process text-only part; see custom model documentation for details
        logger.info_once('Run qwen2_5_omni template')
        processor = self.processor
        # Get multimodal content
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        # We don't use input_ids and attention_mask produced by `processor` because it doesn't produce `labels`.
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        # audio modality
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)  # Find all audio_tokens
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = ((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        # video_audios_mask is used to handle `use_audio_in_video`, distinguishing pure audio(0) from audio in video(1)
        video_audios_mask = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
            else:
                video_audios_mask.append(False)
        video_audios_mask = torch.tensor(video_audios_mask)
        if idx_list:
            # Filter out audio content in videos (will be handled in video section)
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            # Expand multimodal tokens in input_ids
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        # image and video modalities
        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']
                    _get_new_tokens_use_audio_in_video = partial(
                        self._get_new_tokens_use_audio_in_video,
                        video_grid_thw=media_grid_thw,
                        video_second_per_grid=video_second_per_grid,
                        audio_lengths=audio_lengths,
                        video_token_id=token_id,
                        audio_token_id=audio_token_id)
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded.update(media_inputs)  # Add multimodal content
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """This function is typically used to solve the zero2/zero3 hanging issue in mixed model training,
        i.e., some processes have pure text data without passing through vit, while others have image data that passed through vit.
        Here we create dummy_image to solve this.

        This function will be registered in the pre_forward_hook before `model.forward`.
        This function should return input_embeds containing multimodal information.
        """
        if not self.is_training:
            return inputs

        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')
        feature_attention_mask = inputs.get('feature_attention_mask')

        base_model = self.get_base_model(model)
        inputs_embeds = base_model.thinker.model.embed_tokens(input_ids)
        thinker_config = model.config.thinker_config
        # Helper function for handling text/image/video mixed modality data scenarios. (internally creates dummy_image)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.thinker.visual, self.processor,
                                                   thinker_config)
        # Mixed modality data scenarios containing audio
        if input_features is None:
            if is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
                # Note: Due to transformers implementation, the number of passes through audio model layers is related to the number of audios
                # Therefore, zero3 will hang in scenarios where different processes have different numbers of audios (requires modification of transformers code to fix). Use zero2 in this scenario.
                input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
                feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
                audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
                inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return {'inputs_embeds': inputs_embeds}

    def _get_position_ids(self, inputs: Dict[str, Any]):
        """Helper function to get mrope's position_ids"""
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return self._concat_text_position_ids(position_ids)  # First dimension is text_position_ids

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """Passed to dataloader's `collate_fn`"""
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            # padding_free/packing scenarios will handle position_ids in packing_row.
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            # Create `packed_seq_params` to support padding_free/packing & flash-attn
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            # https://github.com/huggingface/transformers/pull/40194
            res.update(get_packed_seq_params(text_position_ids))
        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle multimodal part in `_data_collator` function. (This function is compatible with padding_free/packing)"""
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res

    def generate(self, model, *args, **kwargs):
        """`PtEngine` will call template.generate method for text generation; inherit here for customization."""
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(
    TemplateMeta('my_qwen2_5_omni', prefix=[], prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
                 chat_sep=['<|im_end|>\n'], suffix=['<|im_end|>'],
                 system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
                 default_system='You are a helpful assistant.', stop_words=['<|endoftext|>'],
                 agent_template='hermes',
                 template_cls=Qwen2_5OmniTemplate))

if __name__ == '__main__':
    # Test and debug
    model, processor = get_model_tokenizer('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni')
    template = get_template('my_qwen2_5_omni', processor)
    data = {
        'messages': [
            {'role': 'user', 'content': 'Describe the video<video> and image<image> content.'},
            {'role': 'assistant', 'content': 'A child and a cat.'},
        ],
        'videos': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
        'images': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
    }
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
```

## Inference Alignment

Next, you need to align inference between PtEngine and transformers. Typically you need to align `input_ids` and output content. You can write the following test function:

```python
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from modelscope import snapshot_download
from swift.llm import PtEngine, InferRequest, RequestConfig
import requests

def infer_hf():
    model_dir = snapshot_download('Qwen/Qwen2.5-Omni-7B')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto", attn_implementation='flash_attention_2')
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    # Use decord to read video (url not yet supported)
    resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
    with open('_baby.mp4', 'wb') as f:
        f.write(resp.content)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "_baby.mp4"},
                {"type": "image", "image": "http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"},
                {"type": "text", "text": "Describe the video and image."},
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = False
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                       use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, thinker_do_sample=False,
                              return_audio=False)
    text = processor.batch_decode(text_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return inputs['input_ids'][0].tolist(), text[0]

def test_my_qwen2_5_omni():
    engine = PtEngine('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni', attn_impl='flash_attention_2')
    infer_request = InferRequest(messages=[{
        "role": "user",
        "content": "<video><image>Describe the video and image.",
    }],
        videos=["https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4"],
        images=["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"],
    )
    request_config = RequestConfig(temperature=0, max_tokens=512)
    input_ids = engine.default_template.encode(infer_request)['input_ids']
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    return input_ids, resp


if __name__ == '__main__':
    # Enable debug mode, will print input_ids and generate_ids from `PtEngine.infer`
    os.environ['SWIFT_DEBUG'] = '1'
    input_ids_hf, response_hf = infer_hf()
    input_ids_swift, response_swift = test_my_qwen2_5_omni()
    # Test input_ids and response alignment
    assert input_ids_hf == input_ids_swift
    assert response_hf == response_swift
```


## Start Training

Train using Python code, which is usually easier to debug:


```python
from swift.llm import sft_main, TrainArguments
import os
if __name__ == '__main__':
    os.environ['MAX_PIXELS'] = '1003520'
    sft_main(TrainArguments(
        model='Qwen/Qwen2.5-Omni-7B',
        dataset='AI-ModelScope/LaTeX_OCR#5000',
        model_type='my_qwen2_5_omni',
        template='my_qwen2_5_omni',
        load_from_cache_file=True,
        split_dataset_ratio=0.01,
        train_type='lora',
        torch_dtype='bfloat16',
        attn_impl='flash_attn',
        padding_free=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-4,
        lora_rank=8,
        lora_alpha=32,
        target_modules='all-linear',
        freeze_vit=True,
        freeze_aligner=True,
        gradient_accumulation_steps=1,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        max_length=2048,
        output_dir='output',
        warmup_ratio=0.05,
        dataloader_num_workers=4,
        dataset_num_proc=1,
    ))
```

Train using command line:

```shell
# 4 * 35GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --model_type my_qwen2_5_omni \
    --template my_qwen2_5_omni \
    --custom_register_path 'examples/custom/my_qwen2_5_omni/my_register.py' \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#2000' \
              'swift/VideoChatGPT:all#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_free true \
    --packing true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 1 \
    --deepspeed zero2
```

Perform inference on the validation set after training: (Environment variables should align with training)

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048 \
    --load_data_args true
```

Use the following command to push training weights to Modelscope:

```shell
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>'
```
