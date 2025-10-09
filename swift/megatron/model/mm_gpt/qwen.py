# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args
from PIL import Image

from swift.llm import ModelType, Template
from swift.utils import get_env_args
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model
    if hasattr(language_model, 'language_model'):
        language_model = language_model.language_model
    visual = hf_model.visual if hasattr(hf_model, 'visual') else hf_model.model.visual
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.visual.load_state_dict(visual.state_dict())


def convert_mcore2hf_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model
    if hasattr(language_model, 'language_model'):
        language_model = language_model.language_model
    visual = hf_model.visual if hasattr(hf_model, 'visual') else hf_model.model.visual
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = hf_model.score.weight if args.task_type == 'seq_cls' else hf_model.lm_head.weight
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    visual.load_state_dict(mg_model.visual.visual.state_dict())


class Qwen2_5VL_Vit(HuggingFaceModule):
    module_mapping = {'visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']
    version = 'v2_5'

    def __init__(self, config):
        if self.version == 'v2_5':
            try:
                from transformers.models.qwen2_5_vl import Qwen2_5_VLTextModel
            except ImportError:
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel as Qwen2_5_VLTextModel
            ignore_init_model_cls = Qwen2_5_VLTextModel
        elif self.version == 'v2':
            try:
                from transformers.models.qwen2_vl import Qwen2VLTextModel
            except ImportError:
                from transformers.models.qwen2_vl import Qwen2VLModel as Qwen2VLTextModel
            ignore_init_model_cls = Qwen2VLTextModel
        super().__init__(config, ignore_init_model_cls)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.model_config)


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen2_5_vl, [
            ModelType.qwen2_5_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen2_5_vl,
        convert_mcore2hf=convert_mcore2hf_qwen2_5_vl,
        visual_cls=Qwen2_5VL_Vit))


class Qwen2VL_Vit(Qwen2_5VL_Vit):
    version = 'v2'


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen2_vl, [
            ModelType.qwen2_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen2_5_vl,
        convert_mcore2hf=convert_mcore2hf_qwen2_5_vl,
        visual_cls=Qwen2VL_Vit))


class Qwen2_5Omni_Vit(HuggingFaceModule):
    module_mapping = {
        'thinker': 'thinker',
    }
    _vision_tower = ['thinker.audio_tower', 'thinker.visual']
    _aligner = ['thinker.audio_tower.proj', 'thinker.visual.merger']

    def __init__(self, config):
        from transformers.models.qwen2_5_omni import (Qwen2_5OmniThinkerTextModel,
                                                      Qwen2_5OmniTalkerForConditionalGeneration,
                                                      Qwen2_5OmniToken2WavModel)
        super().__init__(
            config, [Qwen2_5OmniThinkerTextModel, Qwen2_5OmniTalkerForConditionalGeneration, Qwen2_5OmniToken2WavModel])

    def prepare_model(self, hf_model):
        del self.thinker.model
        del self.thinker.lm_head

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        thinker_config = self.model_config.thinker_config
        inputs_embeds = Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.thinker.visual, self.processor,
                                                       thinker_config)
        input_ids = kwargs['input_ids']
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')

        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.thinker.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_embeds = self.thinker.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = self.thinker.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return inputs_embeds


def convert_hf2mcore_qwen2_5_omni(hf_model, mg_model):
    language_model = hf_model.thinker.model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.thinker.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.thinker.visual.load_state_dict(hf_model.thinker.visual.state_dict())
    mg_model.visual.thinker.audio_tower.load_state_dict(hf_model.thinker.audio_tower.state_dict())


def convert_mcore2hf_qwen2_5_omni(hf_model, mg_model):
    language_model = hf_model.thinker.model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = (
            hf_model.thinker.score.weight if args.task_type == 'seq_cls' else hf_model.thinker.lm_head.weight)
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    hf_model.thinker.visual.load_state_dict(mg_model.visual.thinker.visual.state_dict())
    hf_model.thinker.audio_tower.load_state_dict(mg_model.visual.thinker.audio_tower.state_dict())


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen2_5_omni, [
            ModelType.qwen2_5_omni,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen2_5_omni,
        convert_mcore2hf=convert_mcore2hf_qwen2_5_omni,
        visual_cls=Qwen2_5Omni_Vit))


def convert_hf2mcore_ovis2_5(hf_model, mg_model):
    convert_hf2mcore(hf_model.llm, mg_model.language_model)
    mg_model.visual.visual_tokenizer.load_state_dict(hf_model.visual_tokenizer.state_dict())
    mg_model.visual.vte.load_state_dict(hf_model.vte.state_dict())


def convert_mcore2hf_ovis2_5(hf_model, mg_model):
    convert_mcore2hf(hf_model.llm, mg_model.language_model)
    hf_model.visual_tokenizer.load_state_dict(mg_model.visual.visual_tokenizer.state_dict())
    hf_model.vte.load_state_dict(mg_model.visual.vte.state_dict())


class Ovis2_5Vit(HuggingFaceModule):
    module_mapping = {'visual_tokenizer': 'visual_tokenizer', 'vte': 'vte'}
    _vision_tower = ['visual_tokenizer.vit', 'vte']
    _aligner = ['visual_tokenizer.head']

    def __init__(self, config):
        from transformers.models import Qwen3ForCausalLM
        super().__init__(config, Qwen3ForCausalLM)
        self.min_pixels = get_env_args('min_pixels', int, 448 * 448)
        self.max_pixels = get_env_args('max_pixels', int, 1344 * 1792)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values', None)
        grid_thws = kwargs.get('grid_thws')
        INDICATOR_IDS = [-301, -302, -303, -304]
        VISUAL_ATOM_ID = -300
        device = inputs_embeds.device
        visual_indicator_embeds = self.vte(model.indicator_token_indices.to(device=device)).to(
            dtype=inputs_embeds.dtype, device=device)
        inputs_embeds = inputs_embeds.clone()
        for i, indicator_id in enumerate(INDICATOR_IDS):
            inputs_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
        if pixel_values is None:
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                Image.new('RGB', (32, 32), (0, 0, 0)), min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            pixel_values = pixel_values.to(device=inputs_embeds.device)
            grid_thws = grid_thws.to(device=inputs_embeds.device)
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds += visual_embeds.mean() * 0.
        else:
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.ovis2_5, [
            ModelType.ovis2_5,
        ],
        convert_hf2mcore=convert_hf2mcore_ovis2_5,
        convert_mcore2hf=convert_mcore2hf_ovis2_5,
        visual_cls=Ovis2_5Vit))
