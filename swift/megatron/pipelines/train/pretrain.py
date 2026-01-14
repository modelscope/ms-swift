# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronPretrainArguments
from swift.utils import get_logger
from .sft import MegatronSft

logger = get_logger()


class MegatronPretrain(MegatronSft):
    args_class = MegatronPretrainArguments
    args: args_class


def megatron_pretrain_main(args: Optional[Union[List[str], MegatronPretrainArguments]] = None):
    return MegatronPretrain(args).main()
