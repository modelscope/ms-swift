# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .train import megatron_pretrain_main, megatron_rlhf_main, megatron_sft_main
    from .export import megatron_export_main
else:
    _import_structure = {
        'train': ['megatron_pretrain_main', 'megatron_rlhf_main', 'megatron_sft_main'],
        'export': ['megatron_export_main'],
    }
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
