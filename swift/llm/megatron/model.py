import importlib
from typing import Callable, List, Optional

from swift.llm import MODEL_MAPPING

MEGATRON_MODEL_MAPPING = {}


def register_megatron_model(
        model_type_list: List[str],
        convert_module: str,
        model_module: str,  # GPTModel
        config_cls,  # transformer_config_cls
        get_function: Optional[Callable] = None):
    model_info = {
        'convert_module': convert_module,
        'model_module': model_module,
        'config_cls': config_cls,
    }
    if get_function is not None:
        model_info['get_function'] = get_function
        for model_type in model_type_list:
            MEGATRON_MODEL_MAPPING[model_type] = model_info
        return

    def _register_model(get_function: Callable) -> Callable:
        model_info['get_function'] = get_function
        for model_type in model_type_list:
            MEGATRON_MODEL_MAPPING[model_type] = model_info
        return get_function

    return _register_model


qwen1half_model_type = [model_type for model_type in MODEL_MAPPING.keys() if model_type.startswith('qwen1half')]


@register_megatron_model([model_type for model_type in qwen1half_model_type if '32b' not in model_type],
                         'qwen.hf2mcore_qwen1_5_dense_mha', 'qwen1_5', 'QwenTransformerConfig')
@register_megatron_model([model_type for model_type in MODEL_MAPPING.keys() if model_type.startswith('qwen2')],
                         'qwen.hf2mcore_qwen2_dense_and_moe_gqa', 'qwen2', 'Qwen2TransformerConfig')
def get_model_provider(args, gpt_model_cls, config, transformer_layer_spec):

    def model_provider(pre_process=True, post_process=True):
        model = gpt_model_cls(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor)
        return model

    return model_provider


def get_megatron_model_convert(model_type: str):
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    args = get_args()
    model_info = MEGATRON_MODEL_MAPPING[model_type]
    model_module = model_info['model_module']
    config_cls = model_info['config_cls']

    gpt_model_cls = importlib.import_module(f'megatron_patch.model.{model_module}.model').GPTModel
    transformer_config_cls = getattr(
        importlib.import_module(f'megatron_patch.model.{model_module}.transformer_config'), config_cls)
    layer_specs = importlib.import_module(f'megatron_patch.model.{model_module}.layer_specs')
    transformer_layer_spec = layer_specs.get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm,
                                                                                    args.qk_layernorm)
    config = core_transformer_config_from_args(args, transformer_config_cls)
    model_provider = model_info['get_function'](args, gpt_model_cls, config, transformer_layer_spec)
    convert_module = importlib.import_module(f"toolkits.model_checkpoints_convertor.{model_info['convert_module']}")
    return model_provider, convert_module
