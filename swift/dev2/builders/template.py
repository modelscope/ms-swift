# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from swift.dev2.config import TemplateConfig


def build_template(template_config: TemplateConfig, processor: Any, *, task_type: Optional[str] = None) -> Any:
    """TemplateConfig + processor -> a swift Template whose labels are next-token shifted.

    Wraps legacy get_template and derives ``Shifted<LegacyClass>`` from the class it returns, adding
    only dev's twinkle contract (DevMixin); deriving keeps every legacy override (multimodal needs it).
    """
    from swift.dev2.template import shifted_template_class
    from swift.template import get_template

    # legacy Template's vocabulary has no 'delete' (that is a dataset-level length filter) -- the
    # dev default maps onto legacy's own 'raise' default.
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
        # Must be forwarded: legacy _swift_encode reads self.add_non_thinking_prefix, and leaving it
        # out silently took legacy's True default (--add_non_thinking_prefix false was then ignored).
        add_non_thinking_prefix=template_config.add_non_thinking_prefix,
    )
    # Derive in place rather than copy.copy + re-class: nothing needs a separate instance.
    legacy.__class__ = shifted_template_class(type(legacy))
    template = legacy
    # task_type drives legacy's _encode/_embedding_encode and data_collator branches; get_template
    # never sees it here (processor built with load_model=False), so it must be set explicitly.
    if task_type is not None:
        template.task_type = task_type
    template.set_mode('train')
    return template
