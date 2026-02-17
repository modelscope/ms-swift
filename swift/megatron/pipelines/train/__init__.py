# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .pretrain import megatron_pretrain_main
    from .rlhf import megatron_rlhf_main
    from .sft import megatron_sft_main
else:
    _import_structure = {
        'pretrain': ['megatron_pretrain_main'],
        'rlhf': ['megatron_rlhf_main'],
        'sft': ['megatron_sft_main'],
    }
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
