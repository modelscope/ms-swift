# Copyright (c) ModelScope Contributors. All rights reserved.
from .model_config import get_mcore_model_config


def get_mcore_model(args, processor, hf_config):
    from mcore_bridge import get_mcore_model as _get_mcore_model
    config = get_mcore_model_config(args, processor, hf_config)
    models = _get_mcore_model(config)

    return models
