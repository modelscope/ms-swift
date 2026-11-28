# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import nullcontext

from .base import ModelLoader, register_model


@register_model
class Ling3Loader(ModelLoader):
    model_type = "ling3"
    model_cls = "transformers:AutoModelForCausalLM"
    trust_remote_code = True
    architectures = ["BailingMoeV3ForCausalLM"]
    template = "ling3"
    is_moe = True
    mcore_model_type = "bailing_hybrid"
    requires = ["transformers>=4.57.1,<6", "einops", "flash-linear-attention"]
    models = ["inclusionAI/Ling-3.0-tiny", "inclusionAI/Ling-3.0-flash"]

    def process_config(self, config):
        import transformers.utils.import_utils as import_utils

        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = lambda: True
        if getattr(config, "rope_parameters", None) is None:
            config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
        return config

    def _construction_context(self, model_dir, config, kwargs):
        from transformers import __version__

        from ._ling3_compat import is_ling3_tiny
        from ._ling3_rope import ling3_rope_construction, transformers_compatibility

        tiny = is_ling3_tiny(config)
        path = transformers_compatibility(__version__, warn=True) if tiny else None
        rope = getattr(config, "rope_scaling", None)
        if isinstance(rope, dict) and "factor" not in rope:
            if not tiny or (path == "tf4" and rope.get("rope_type", "default") == "default"
                            and rope.get("type", "default") == "default"):
                config.rope_scaling = dict(rope, factor=1.0, rope_type="linear")
        return ling3_rope_construction(model_dir, config, kwargs) if tiny else nullcontext(None)

    def build_model(self, model_dir, config, processor, **kwargs):
        with self._construction_context(model_dir, config, kwargs) as finalize:
            model = super().build_model(model_dir, config, processor, **kwargs)
            if finalize is not None:
                finalize(model)
        return model

    def build_model_from_config(self, config, processor, **kwargs):
        self._apply_trust_remote_code(kwargs)
        with self._construction_context(self.model_info.model_dir, config, kwargs):
            model = self.resolve_model_cls().from_config(config, **kwargs)
            # Meta peers rebuild RoPE when FSDP materializes their buffers via _apply.
            model._tied_weights_keys = {}
        return model

    @staticmethod
    def _extend_mtp_cache(model):
        import sys
        from transformers.cache_utils import DynamicCache

        class MtpDynamicCache(DynamicCache):
            def update(self, key_states, value_states, layer_idx, *args, **kwargs):
                if hasattr(self, 'layers'):
                    from transformers.cache_utils import DynamicLayer

                    while len(self.layers) <= layer_idx:
                        self.layers.append(DynamicLayer())
                return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # Only the loaded remote-code modules use this cache; other families retain the native class.
        for module in model.modules():
            namespace = sys.modules[type(module).__module__]
            if getattr(namespace, 'DynamicCache', None) is DynamicCache:
                namespace.DynamicCache = MtpDynamicCache

    def process_model(self, model):
        from ._ling3_compat import install_ling3_compat, is_ling3_tiny, patch_ling3_packing

        if getattr(model.config, 'num_nextn_predict_layers', 0):
            self._extend_mtp_cache(model)
        if is_ling3_tiny(model.config):
            if not getattr(model.config, "num_nextn_predict_layers", 0):
                install_ling3_compat(model)
            patch_ling3_packing(model)
        return model
