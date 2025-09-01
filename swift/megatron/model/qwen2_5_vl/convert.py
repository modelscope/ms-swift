from megatron.training import get_args

from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf


def convert_hf2mcore_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    args = get_args()
    mg_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_model, language_model, layer_idx)
    mg_model.visual.model.load_state_dict(hf_model.model.visual.state_dict())


def convert_mcore2hf_qwen2_5_vl(hf_model, mg_model):
    language_model = hf_model.model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        hf_model.lm_head.weight.data.copy_(mg_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_model, language_model, layer_idx)
    hf_model.model.visual.load_state_dict(mg_model.visual.model.state_dict())
