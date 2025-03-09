# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.llm.train import SwiftSft
from swift.utils import get_logger
from ..argument import MegatronTrainArguments

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronTrainArguments
    args: args_class


def megatron_sft_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronSft(args).main()
