"""build_template: TemplateConfig + processor -> dev Template (VL subclass via mapping)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swift.dev.configs import TemplateConfig


def build_template(template_config: TemplateConfig, processor: Any) -> Any:
    """TemplateConfig + processor -> a template whose labels are next-token shifted.

    Wraps the legacy get_template (which resolves template_type + injects
    max_length/system/truncation/loss_scale) and adds dev's label convention on top. Two shapes,
    chosen by `legacy_encode`:

      True (default) -- KEEP the legacy class and derive `Shifted<LegacyClass>` from it, so the
        family's own `_encode`/`replace_tag`/`_data_collator`/`_get_position_ids`/... all still
        dispatch and only `encode` is added.
      False -- re-class into dev's own Template subclass (`PROCESSOR_TEMPLATE_MAPPING` by
        template_type), i.e. dev's chat-template encode rewrite. Opt-in: see TemplateConfig.

    Why the default is not "re-class + delegate _encode to legacy": re-classing REPLACES the legacy
    class, so every method the legacy subclass overrode stops dispatching -- measured on Qwen3.5,
    14 of them -- and `super()._encode()` lands on the BASE legacy `_encode` instead of the family's.
    Two of those overrides are on legacy's OWN call path and so are not neutralised by delegating
    `_encode` at all: `replace_tag` (base.py:1000/1035, inside `_swift_encode`), whose dev version
    deliberately does no media preprocessing while legacy's `fetch_image`s in place, and
    `Gemma3VisionTemplate._swift_prepare_inputs` (base.py:1492). Text happens to survive that
    (`qwen2_5`/`qwen3` ARE the base class, so nothing is dropped); multimodal does not.
    """
    from swift.dev.template import PROCESSOR_TEMPLATE_MAPPING, shifted_template_class
    from swift.template import get_template

    # follow legacy, TODO: refactor
    truncation_strategy = template_config.truncation_strategy
    if truncation_strategy in (None, 'delete'):
        truncation_strategy = 'raise'

    legacy = get_template(
        processor,
        default_system=template_config.system,
        max_length=template_config.max_length,
        template_type=template_config.template,
        truncation_strategy=truncation_strategy,
        loss_scale=template_config.loss_scale,
        padding_free=template_config.padding_free,
        padding_side=template_config.padding_side,
        sequence_parallel_size=template_config.sequence_parallel_size,
        # Must be forwarded: the dev Template reuses legacy _swift_encode, whose
        # _add_non_thinking_prefix branch reads self.add_non_thinking_prefix (template/base.py:1279).
        # Leaving it out silently took legacy's True default, so `--add_non_thinking_prefix false`
        # was honoured by legacy but ignored here -- dev then prepended an empty
        # '<think>\n\n</think>\n\n' (4 tokens) to every assistant turn and no legacy-vs-dev loss
        # comparison could line up.
        add_non_thinking_prefix=template_config.add_non_thinking_prefix,
    )
    if template_config.legacy_encode:
        # Derive in place rather than copy.copy + re-class: there is nothing to preserve a separate
        # instance for, and the derived class keeps the legacy one as its base.
        legacy.__class__ = shifted_template_class(type(legacy))
        template = legacy
    else:
        from swift.dev.template import Template as DevTemplate
        tt = legacy.template_meta.template_type
        cls = PROCESSOR_TEMPLATE_MAPPING.get(tt, DevTemplate)
        template = cls.from_template(legacy)
    template.set_mode('train')
    return template
