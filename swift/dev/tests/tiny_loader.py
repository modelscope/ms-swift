# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the tiny checkpoint through dev's own family loaders instead of a bare ``AutoModel``.

The dev-only half of ``tiny.py``, kept separate because that module must stay importable by legacy
swift, which has no :class:`~swift.dev.model.loader.ModelLoader`. It plugs in as
``TinyModel.build(builder=loader_builder(model_type))``.

What the loader buys over naming a config class: the family already declares which transformers
classes it loads with, and the real ``config.json`` -- nested text/vision sub-configs included -- is
the starting point, so a row in a model table says "``qwen3_moe``, this size" instead of restating an
architecture and its MoE geometry. Only the size fields are overwritten; everything else about the
checkpoint stays as the family ships it, which is why this is a *subclass* of the registered loader
rather than a re-implementation of it -- a family that overrides ``build_config`` /
``build_processor`` / ``process_model`` keeps doing so under the tiny sizes.
"""
from __future__ import annotations
from typing import Any, Dict, Type

from transformers import PretrainedConfig

from swift.dev.model.loader import ModelInfo, ModelLoader, get_model_loader
from swift.dev.tests.tiny import Builder


def _truncate_per_layer(config: PretrainedConfig, old_layers: int, new_layers: int) -> None:
    """Cut every per-layer list down to the new layer count.

    transformers>=4.54 configs carry ``layer_types`` -- one entry per layer -- and validate its length
    against ``num_hidden_layers`` when the config is *saved*, so shrinking the count alone fails late
    with "num_hidden_layers (4) must be equal to the number of layer_types (24)". Any list as long as
    the ORIGINAL layer count is such a per-layer list (hybrid families add their own), and the leading
    slice is the right cut: it keeps the family's interleave pattern -- full/sliding attention,
    dense-then-MoE -- as it starts at layer 0, rather than inventing a uniform stack.
    """
    for key, value in vars(config).items():
        if isinstance(value, list) and len(value) == old_layers:
            setattr(config, key, value[:new_layers])


def shrink(config: PretrainedConfig, dims: Dict[str, Any]) -> PretrainedConfig:
    """Overwrite the size fields on ``config`` and on every sub-config that declares them.

    Only fields the config *already* has are written: ``num_experts`` must not be grafted onto a
    dense config (the saved ``config.json`` would then advertise a MoE the model does not have), and
    a vision tower has no ``vocab_size``. Sub-configs are shrunk too -- a VL checkpoint keeps its
    layer count in ``text_config`` / ``vision_config``, so stopping at the top level would leave a
    full-size tower behind and defeat the whole point of building tiny.
    """
    old_layers = getattr(config, 'num_hidden_layers', None)
    for key, value in dims.items():
        if hasattr(config, key):
            setattr(config, key, value)
    new_layers = getattr(config, 'num_hidden_layers', None)
    if old_layers and new_layers and new_layers < old_layers:
        _truncate_per_layer(config, old_layers, new_layers)
    for value in vars(config).values():
        if isinstance(value, PretrainedConfig):
            shrink(value, dims)
    return config


def tiny_loader_cls(model_type: str, dims: Dict[str, Any]) -> Type[ModelLoader]:
    """The registered loader for ``model_type``, subclassed to build tiny and from nothing.

    Two hooks change and no more: ``build_config`` shrinks whatever the family resolved, and
    ``build_model`` swaps ``from_pretrained`` for ``from_config`` because the snapshot holds config and
    tokenizer files only -- there are no weights on disk to read, and random init is the point.
    """
    base = get_model_loader(model_type)

    class TinyLoader(base):

        def build_config(self, model_dir: str, **kwargs) -> PretrainedConfig:
            return shrink(super().build_config(model_dir, **kwargs), dims)

        def build_model(self, model_dir: str, config: PretrainedConfig, processor, **kwargs):
            return self.resolve_model_cls().from_config(config, **kwargs)

    TinyLoader.__name__ = f'Tiny{base.__name__}'
    TinyLoader.__qualname__ = TinyLoader.__name__
    return TinyLoader


def loader_builder(model_type: str) -> Builder:
    """A :data:`~swift.dev.tests.tiny.Builder` that builds ``model_type``'s own architecture, shrunk.

    Walks the same sequence the real load path does -- config, processor, model, each through the
    family's hooks -- so a family whose ``process_config`` fixes up its config, or whose
    ``build_model`` needs the processor, is exercised here rather than bypassed.
    """

    def _build(snapshot_dir: str, dims: Dict[str, Any]):
        model_info = ModelInfo(model_type=model_type, model_dir=snapshot_dir, torch_dtype=dims.get('dtype'))
        loader = tiny_loader_cls(model_type, dims)(model_info)
        config = loader.process_config(loader.build_config(snapshot_dir))
        processor = loader.build_processor(snapshot_dir, config)
        return loader.process_model(loader.build_model(snapshot_dir, config, processor))

    return _build
