import torch
from megatron.training import get_args, get_tokenizer

from swift.llm import ModelType, Template
from ..constant import MegatronModelType
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
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
        hf_model.lm_head.weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)
    visual.load_state_dict(mg_model.visual.visual.state_dict())


class Qwen2_5VL_Vit(HuggingFaceModule):
    module_mapping = {'visual': 'visual'}
    vision_tower = ['visual']
    aligner = ['visual.merger']
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
    vision_tower = ['thinker.audio_tower', 'thinker.visual']
    aligner = ['thinker.audio_tower.proj', 'thinker.visual.merger']

    def __init__(self, config):
        from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerTextModel
        super().__init__(config, [Qwen2_5OmniThinkerTextModel])

    def prepare_model(self, hf_model):
        self.thinker.model = None
        self.thinker.lm_head = None

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
        hf_model.thinker.lm_head.weight.data.copy_(mg_language_model.output_layer.weight)
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
