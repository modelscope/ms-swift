# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .animatediff import animatediff_sft, animatediff_main
    from .animatediff_infer import animatediff_infer, animatediff_infer_main
    from .diffusers import train_text_to_image, train_text_to_image_lora, train_text_to_image_lora_sdxl, \
        train_text_to_image_sdxl, infer_text_to_image, infer_text_to_image_lora, infer_text_to_image_sdxl, \
        infer_text_to_image_lora_sdxl, train_controlnet, train_controlnet_sdxl, train_dreambooth, \
        train_dreambooth_lora, train_dreambooth_lora_sdxl, infer_controlnet, infer_controlnet_sdxl, \
        infer_dreambooth, infer_dreambooth_lora, infer_dreambooth_lora_sdxl
    from .utils import AnimateDiffArguments, AnimateDiffInferArguments
else:
    _import_structure = {
        'animatediff': ['animatediff_sft', 'animatediff_main'],
        'animatediff_infer': ['animatediff_infer', 'animatediff_infer_main'],
        'diffusers': [
            'train_text_to_image', 'train_text_to_image_lora', 'train_text_to_image_lora_sdxl',
            'train_text_to_image_sdxl', 'infer_text_to_image', 'infer_text_to_image_lora', 'infer_text_to_image_sdxl',
            'infer_text_to_image_lora_sdxl', 'train_controlnet', 'train_controlnet_sdxl', 'train_dreambooth',
            'train_dreambooth_lora', 'train_dreambooth_lora_sdxl', 'infer_controlnet', 'infer_controlnet_sdxl',
            'infer_dreambooth', 'infer_dreambooth_lora', 'infer_dreambooth_lora_sdxl'
        ],
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
