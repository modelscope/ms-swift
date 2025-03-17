# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from ..config import load_config


def load_qwen_config(config) -> Dict[str, Any]:
    args_config = load_config(config)
    args_config['swiglu'] = True
    return args_config
