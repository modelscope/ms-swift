# Copyright (c) Alibaba, Inc. and its affiliates.

try:
    from .init import init_megatron_env
    init_megatron_env()
except Exception:
    # allows lint pass.
    raise

from .convert import convert_hf2megatron, convert_megatron2hf
