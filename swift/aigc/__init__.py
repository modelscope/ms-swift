# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .animatediff import animatediff_sft, animatediff_main
    from .animatediff_infer import animatediff_infer, animatediff_infer_main
    from .utils import AnimateDiffArguments, AnimateDiffInferArguments
else:
    _import_structure = {
        'animatediff': ['animatediff_sft', 'animatediff_main'],
        'animatediff_infer': ['animatediff_infer', 'animatediff_infer_main'],
        'utils': ['AnimateDiffArguments', 'AnimateDiffInferArguments'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
