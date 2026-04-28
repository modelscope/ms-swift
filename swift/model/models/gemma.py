# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from PIL import Image
from transformers import PreTrainedModel
from types import MethodType

from swift.template import TemplateType
from swift.utils import is_deepspeed_enabled, to_device
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_to_input_device
from ..register import ModelLoader, SentenceTransformersLoader, register_model


class PaligemmaVisionLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import PaliGemmaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or PaliGemmaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.paligemma,
        [
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
                Model('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
                Model('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-mix-224', 'google/paligemma-3b-mix-224'),
                Model('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-pt-224', 'google/paligemma2-3b-pt-224'),
                Model('AI-ModelScope/paligemma2-3b-pt-448', 'google/paligemma2-3b-pt-448'),
                Model('AI-ModelScope/paligemma2-3b-pt-896', 'google/paligemma2-3b-pt-896'),
                Model('AI-ModelScope/paligemma2-10b-pt-224', 'google/paligemma2-10b-pt-224'),
                Model('AI-ModelScope/paligemma2-10b-pt-448', 'google/paligemma2-10b-pt-448'),
                Model('AI-ModelScope/paligemma2-10b-pt-896', 'google/paligemma2-10b-pt-896'),
                Model('AI-ModelScope/paligemma2-28b-pt-224', 'google/paligemma2-28b-pt-224'),
                Model('AI-ModelScope/paligemma2-28b-pt-448', 'google/paligemma2-28b-pt-448'),
                Model('AI-ModelScope/paligemma2-28b-pt-896', 'google/paligemma2-28b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-ft-docci-448', 'google/paligemma2-3b-ft-docci-448'),
                Model('AI-ModelScope/paligemma2-10b-ft-docci-448', 'google/paligemma2-10b-ft-docci-448'),
            ]),
        ],
        PaligemmaVisionLoader,
        template=TemplateType.paligemma,
        architectures=['PaliGemmaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.41'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma,
        [
            ModelGroup([
                Model('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
                Model('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
                Model('AI-ModelScope/gemma-7b', 'google/gemma-7b'),
                Model('AI-ModelScope/gemma-7b-it', 'google/gemma-7b-it'),
            ], ),
        ],
        template=TemplateType.gemma,
        architectures=['GemmaForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.38'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma2,
        [
            ModelGroup([
                Model('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
                Model('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
                Model('LLM-Research/gemma-2-9b', 'google/gemma-2-9b'),
                Model('LLM-Research/gemma-2-9b-it', 'google/gemma-2-9b-it'),
                Model('LLM-Research/gemma-2-27b', 'google/gemma-2-27b'),
                Model('LLM-Research/gemma-2-27b-it', 'google/gemma-2-27b-it'),
            ], ),
        ],
        template=TemplateType.gemma,
        architectures=['Gemma2ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.42'],
    ))


class Gemma3TextLoader(ModelLoader):

    def get_config(self, model_dir):
        # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_config(model_dir)


register_model(
    ModelMeta(
        LLMModelType.gemma3_text,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-1b-pt', 'google/gemma-3-1b-pt'),
                Model('LLM-Research/gemma-3-1b-it', 'google/gemma-3-1b-it'),
                Model('google/gemma-3-270m', 'google/gemma-3-270m'),
                Model('google/gemma-3-270m-it', 'google/gemma-3-270m-it'),
                Model('google/medgemma-27b-text-it', 'google/medgemma-27b-text-it'),
            ], ),
        ],
        Gemma3TextLoader,
        template=TemplateType.gemma3_text,
        architectures=['Gemma3ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.49'],
    ))


class Gemma3VisionLoader(ModelLoader):

    def get_config(self, model_dir):
        # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Gemma3ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma3ForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.gemma3_vision,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-4b-pt', 'google/gemma-3-4b-pt'),
                Model('LLM-Research/gemma-3-4b-it', 'google/gemma-3-4b-it'),
                Model('LLM-Research/gemma-3-12b-pt', 'google/gemma-3-12b-pt'),
                Model('LLM-Research/gemma-3-12b-it', 'google/gemma-3-12b-it'),
                Model('LLM-Research/gemma-3-27b-pt', 'google/gemma-3-27b-pt'),
                Model('LLM-Research/gemma-3-27b-it', 'google/gemma-3-27b-it'),
                Model('google/medgemma-4b-pt', 'google/medgemma-4b-pt'),
                Model('google/medgemma-4b-it', 'google/medgemma-4b-it'),
                Model('google/medgemma-27b-it', 'google/medgemma-27b-it'),
            ], ),
        ],
        Gemma3VisionLoader,
        template=TemplateType.gemma3_vision,
        architectures=['Gemma3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))


class Gemma3nLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Gemma3nForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma3nForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_to_input_device(model.model.embed_vision)
        patch_output_to_input_device(model.model.embed_audio)
        return model


register_model(
    ModelMeta(
        MLLMModelType.gemma3n,
        [
            ModelGroup([
                Model('google/gemma-3n-E2B', 'google/gemma-3n-E2B'),
                Model('google/gemma-3n-E4B', 'google/gemma-3n-E4B'),
                Model('google/gemma-3n-E2B-it', 'google/gemma-3n-E2B-it'),
                Model('google/gemma-3n-E4B-it', 'google/gemma-3n-E4B-it'),
            ], ),
        ],
        Gemma3nLoader,
        template=TemplateType.gemma3n,
        architectures=['Gemma3nForConditionalGeneration'],
        model_arch=ModelArch.gemma3n,
        requires=['transformers>=4.53.1'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma_emb,
        [
            ModelGroup([
                Model('google/embeddinggemma-300m', 'google/embeddinggemma-300m'),
            ], ),
        ],
        SentenceTransformersLoader,
        template=TemplateType.dummy,
        architectures=['Gemma3TextModel'],
    ))


def _patch_gemma4_forward(model, processor):
    from transformers.models.gemma4.modeling_gemma4 import (Gemma4ModelOutputWithPast, create_causal_mask_mapping,
                                                            create_masks_for_generate, torch_compilable_check)
    if hasattr(model, 'origin_forward'):
        return

    def _forward_dummy_image(model, inputs_embeds):
        images = [Image.new('RGB', (32, 32), (0, 0, 0))]
        image_inputs = processor.image_processor(images=images, return_tensors='pt')
        image_inputs = to_device(image_inputs, inputs_embeds.device)
        dummy_pixel = image_inputs['pixel_values'].to(model.vision_tower.dtype)
        dummy_pos_ids = image_inputs.get('image_position_ids')
        image_features = model.get_image_features(dummy_pixel, dummy_pos_ids, return_dict=True).pooler_output
        inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds

    # transformers 5.6.2
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> Gemma4ModelOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = self.config.text_config.pad_token_id
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if self.config.get_text_config().hidden_size_per_layer_input:
            pad_embedding = self.language_model.embed_tokens.weight[self.config.text_config.pad_token_id, :]
            llm_inputs_embeds = torch.where(multimodal_mask[..., None], pad_embedding.view(1, 1, -1), inputs_embeds)
            per_layer_inputs = self.language_model.get_per_layer_inputs(llm_input_ids, llm_inputs_embeds)
        else:
            per_layer_inputs = None
        # Mixed modality training with both images and videos is not currently supported.
        if is_deepspeed_enabled() and pixel_values is None and pixel_values_videos is None:
            inputs_embeds = _forward_dummy_image(self, inputs_embeds)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f'Image features and image tokens do not match, tokens: {n_image_tokens}, features:'
                f' {image_features.shape[0]}',
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device))

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_position_ids, return_dict=True).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_video_tokens = video_mask.sum()
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[video_mask].numel() == video_features.numel(),
                f'Video features and video tokens do not match, tokens: {n_video_tokens}, features:'
                f' {video_features.shape[0]}',
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device), video_features.to(inputs_embeds.device))

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask  # True = valid

            # Strip padding tokens: only keep real (non-padding) audio soft tokens.
            # audio_mask_from_encoder is True for valid positions, False for padding tokens.
            # This mirrors the vision encoder's padding stripping (see Gemma4VisionEncoder.forward).
            audio_features = audio_features[audio_mask_from_encoder]

            n_audio_tokens = audio_mask.sum()
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[audio_mask].numel() == audio_features.numel(),
                f'Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:'
                f' {audio_features.shape[0] * audio_features.shape[1]}',
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device))
        elif is_deepspeed_enabled() and self.audio_tower is not None:
            feature_size = processor.feature_extractor.feature_size
            dummy_features = input_ids.new_zeros([1, 128, feature_size], dtype=self.audio_tower.dtype)
            dummy_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_output = self.get_audio_features(dummy_features, dummy_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            inputs_embeds = inputs_embeds + audio_features.mean() * 0.

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if self.config.get_text_config().use_bidirectional_attention == 'vision':
                # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
                causal_mask_mapping = create_causal_mask_mapping(
                    self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                )
            else:
                # Smaller Gemma models use a conventional casual attention mask
                causal_mask_mapping = create_masks_for_generate(
                    self.config,
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    position_ids,
                )
        kwargs.pop('return_dict', None)
        outputs = self.language_model(
            per_layer_inputs=per_layer_inputs,
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        return Gemma4ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    model.origin_forward = model.forward
    model.forward = MethodType(forward, model)


class Gemma4Loader(ModelLoader):

    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Gemma4ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma4ForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_gemma4_forward(model.model, processor)
        return model


register_model(
    ModelMeta(
        MLLMModelType.gemma4,
        [
            ModelGroup([
                Model('google/gemma-4-E2B', 'google/gemma-4-E2B'),
                Model('google/gemma-4-E2B-it', 'google/gemma-4-E2B-it'),
                Model('google/gemma-4-E4B', 'google/gemma-4-E4B'),
                Model('google/gemma-4-E4B-it', 'google/gemma-4-E4B-it'),
            ],
                       template=TemplateType.gemma4_nothinking),
            ModelGroup([
                Model('google/gemma-4-31B', 'google/gemma-4-31B'),
                Model('google/gemma-4-31B-it', 'google/gemma-4-31B-it'),
                Model('google/gemma-4-26B-A4B', 'google/gemma-4-26B-A4B'),
                Model('google/gemma-4-26B-A4B-it', 'google/gemma-4-26B-A4B-it'),
            ],
                       template=TemplateType.gemma4),
        ],
        Gemma4Loader,
        architectures=['Gemma4ForConditionalGeneration'],
        model_arch=ModelArch.gemma3n,
        requires=['transformers>=4.53'],
    ))
