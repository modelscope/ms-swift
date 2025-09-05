# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional, Union

import torch

from swift.llm import TEMPLATE_MAPPING, ExportArguments
from swift.llm.train import SwiftSft
from swift.utils import get_logger

logger = get_logger()


class ExportCachedDataset(SwiftSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        super(SwiftSft, self).__init__(args)
        self.train_msg = {}  # dummy
        template_cls = TEMPLATE_MAPPING[args.template].template_cls
        if template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self._prepare_model_tokenizer(**kwargs)
        self._prepare_template()

    def main(self):
        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        self._show_dataset(train_dataset, val_dataset)
        train_dataset.save_to_disk(os.path.join(self.args.output_dir, 'train'))
        if val_dataset is not None:
            val_dataset.save_to_disk(os.path.join(self.args.output_dir, 'val'))
        logger.info(f'Dataset saved to `{self.args.output_dir}`')


def export_cached_dataset(args: Optional[Union[List[str], ExportArguments]] = None):
    return ExportCachedDataset(args).main()
