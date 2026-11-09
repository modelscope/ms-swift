# Copyright (c) ModelScope Contributors. All rights reserved.
import os


def prepare_entropack_model(args, model):
    if getattr(model, 'model_info', None) is None or model.model_info.quant_method != 'entropack':
        return model

    from transformers.integrations.entropack import (get_entropack_fsdp_ignored_modules_pattern,
                                                     prepare_entropack_model_for_ddp)

    prepare_entropack_model_for_ddp(model)
    if getattr(args, 'fsdp', None):
        pattern = get_entropack_fsdp_ignored_modules_pattern(model)
        existing = os.environ.get('FSDP_IGNORED_MODULES')
        os.environ['FSDP_IGNORED_MODULES'] = pattern if not existing else f'(?:{existing})|(?:{pattern})'
    return model
