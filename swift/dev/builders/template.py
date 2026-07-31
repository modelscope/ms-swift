"""build_template: TemplateConfig + processor -> dev Template (VL subclass via mapping)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swift.dev.configs import TemplateConfig


def build_template(template_config: TemplateConfig, processor: Any) -> Any:
    """TemplateConfig + processor -> a swift Template whose labels are next-token shifted.

    Wraps the legacy get_template (which resolves template_type + injects
    max_length/system/truncation/loss_scale) and derives `Shifted<LegacyClass>` from the class it
    returns, adding only dev's twinkle contract (`DevMixin`: the label convention + batch_encode).

    Deriving rather than re-classing is the whole point: overwriting `__class__` drops every method
    the legacy subclass overrode -- measured on Qwen3.5: 14, including `_encode`, `replace_tag`,
    `_data_collator`, `_get_position_ids`, `packing_row` -- and silently routes `super()._encode()`
    to the BASE legacy `_encode` instead of the family's. Two of those overrides sit on legacy's OWN
    call path, so they are not neutralised by delegating `_encode`: `replace_tag` (base.py, inside
    `_swift_encode`), whose media preprocessing `fetch_image`s in place, and
    `Gemma3VisionTemplate._swift_prepare_inputs`. Text survives re-classing (`qwen2_5`/`qwen3` ARE the
    base class); multimodal does not.

    The returned template is used by BOTH the dataset (encode/lengths/packing) and the model (twinkle
    calls `batch_encode`), so one implementation produces the training tokens.
    """
    from swift.dev.template import shifted_template_class
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
    # Derive in place rather than copy.copy + re-class: there is nothing to preserve a separate
    # instance for, and the derived class keeps the legacy one as its base.
    legacy.__class__ = shifted_template_class(type(legacy))
    template = legacy
    template.set_mode('train')
    return template
