# Copyright (c) ModelScope Contributors. All rights reserved.

import sys
import transformers
from packaging import version
from transformers import PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()

transformers_5 = version.parse(transformers.__version__) >= version.parse('5.0.0.dev')


class Spark2_5Loader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        # `modeling_spark.py` calls `eager_attention_forward` directly and declares neither
        # sdpa nor flash_attn support, so any other implementation would be rejected by
        # `from_pretrained`.
        if self.attn_impl not in {None, 'eager'}:
            logger.warning(f'Spark-X2.5 only implements eager attention, '
                           f'ignoring attn_impl: "{self.attn_impl}".')
        self.attn_impl = 'eager'
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        self._patch_remote_code(model_dir)
        return super().get_model(model_dir, *args, **kwargs)

    @staticmethod
    def _patch_remote_code(model_dir: str) -> None:
        """Make `modeling_spark.py`, which targets transformers 4.57, importable under transformers>=5.

        Patching the class/module before `from_pretrained` resolves them keeps the loading path
        (and therefore the remote-code saving) unchanged.
        """
        if not transformers_5:
            return
        model_cls = get_class_from_dynamic_module('modeling_spark.Spark2_5ForCausalLM', model_dir)
        modeling_module = sys.modules[model_cls.__module__]
        if getattr(modeling_module, '_swift_patched', False):
            return
        modeling_module._swift_patched = True

        # transformers>=5 expects a {tied_weight: source_weight} mapping, and `post_init` calls
        # `.keys()` on it, so the 4.x list form cannot even be constructed.
        if isinstance(model_cls._tied_weights_keys, list):
            # `Spark2_5Model` names its embedding `embedding`, not `embed_tokens`.
            model_cls._tied_weights_keys = {'lm_head.weight': 'model.embedding.weight'}

        # transformers>=5 renamed `input_embeds` to `inputs_embeds` and dropped `cache_position`.
        for name in ['create_causal_mask', 'create_sliding_window_causal_mask']:
            setattr(modeling_module, name, Spark2_5Loader._compat_mask_fn(getattr(modeling_module, name)))

    @staticmethod
    def _compat_mask_fn(mask_fn):

        def new_mask_fn(*args, **kwargs):
            if 'input_embeds' in kwargs:
                kwargs['inputs_embeds'] = kwargs.pop('input_embeds')
            kwargs.pop('cache_position', None)
            return mask_fn(*args, **kwargs)

        return new_mask_fn


register_model(
    ModelMeta(
        LLMModelType.spark2_5,
        [
            ModelGroup([
                Model('XHToken/Spark-X2.5-1.7B-Base'),
                Model('XHToken/Spark-X2.5-1.7B'),
                Model('XHToken/Spark-X2.5-4B-Base'),
                Model('XHToken/Spark-X2.5-4B'),
            ]),
        ],
        Spark2_5Loader,
        template=TemplateType.spark2_5,
        architectures=['Spark2_5ForCausalLM'],
        # `modeling_spark.py` imports `transformers.masking_utils.create_sliding_window_causal_mask`
        # and `transformers.utils.TransformersKwargs`; config.json was exported by 4.57.1.
        requires=['transformers>=4.57'],
    ))
