# Copyright (c) Alibaba, Inc. and its affiliates.
from .argument import MegatronArguments
from .convert import convert_hf_to_megatron, convert_megatron_to_hf
from .model import MEGATRON_MODEL_MAPPING, get_megatron_model_convert, register_megatron_model
from .utils import forward_step, init_megatron_env, patch_megatron, train_valid_test_datasets_provider

init_megatron_env()
