# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import List, Union

from megatron.core.enums import ModelType
from megatron.training import pretrain

from swift.llm.train import SwiftSft
from swift.utils import get_logger
from ..argument import MegatronTrainArguments
from ..utils import patch_megatron
from .patcher import dummy_func, patch_megatron_dataset, patch_training_log
from .utils import forward_step

logger = get_logger()


class MegatronSft(SwiftSft):
    args_class = MegatronTrainArguments
    args: args_class

    def __init__(self, args: Union[List[str], MegatronTrainArguments, None] = None) -> None:
        super().__init__(args)
        _, self.processor = args.get_model_processor(load_model=False)
        patch_megatron(self.processor)
        self._prepare_template()
        self.args.save_args()

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        self._save_val_dataset(args.save, val_dataset)
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)

        megatron_model_meta = args.megatron_model_meta
        model_provider = megatron_model_meta.get_model_provider()
        train_valid_test_datasets_provider = dummy_func
        dummy_func.is_distributed = True
        with patch_training_log(), patch_megatron_dataset(self.template, train_dataset, val_dataset):
            pretrain(
                train_valid_test_datasets_provider,
                model_provider,
                ModelType.encoder_or_decoder,
                forward_step,
                args_defaults=args.extra_args)


def megatron_sft_main(args: Union[List[str], MegatronTrainArguments, None] = None):
    return MegatronSft(args).main()
