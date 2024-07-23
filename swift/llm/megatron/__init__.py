# Copyright (c) Alibaba, Inc. and its affiliates.
from .argument import MegatronArguments
from .convert import convert_hf_to_megatron, convert_megatron_to_hf, model_provider
from .utils import forward_step, get_model_seires, init_megatron_env, patch_megatron, train_valid_test_datasets_provider

init_megatron_env()
