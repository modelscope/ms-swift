# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.utils import get_env_args


def prepare_minicpm_ocr_model(args, model):
    if args.model_type == 'minicpm_ocr':
        template_cls = getattr(getattr(args, 'template_meta', None), 'template_cls', None)
        default_mode = getattr(template_cls, 'default_downsample_mode', '4x')
        if get_env_args('downsample_mode', str, default_mode) == '4x':
            for name, module in model.named_modules():
                if name.endswith('vision_tower.vit_merger'):
                    module.requires_grad_(False)
    return model
