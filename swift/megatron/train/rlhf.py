# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union

from megatron.core.enums import ModelType
from megatron.training import pretrain

from swift.utils import get_logger, is_master, plot_images
from ..argument import MegatronRLHFArguments
from ..utils import patch_megatron_tokenizer
from .patcher import patch_megatron_data_collator, patch_training_log
from .sft import MegatronSft
from .utils import build_streaming_dataloader, forward_step, get_swift_datasets_provider

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()
