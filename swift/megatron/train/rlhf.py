# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union
from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()
