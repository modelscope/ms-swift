# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import List, Optional, Union

import torch

from swift.arguments import ExportArguments
from swift.utils import get_logger
from ..train import SwiftSft

logger = get_logger()


class ExportCachedDataset(SwiftSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        super(SwiftSft, self).__init__(args)
        args = self.args
        self.train_msg = {}  # dummy
        template_cls = args.template_meta.template_cls
        if template_cls and template_cls.use_model:
            kwargs = {'return_dummy_model': True}
        else:
            kwargs = {'load_model': False}
        with torch.device('meta'):
            self._prepare_model_tokenizer(**kwargs)
        self._prepare_template()
        self.template.set_mode(args.template_mode)

    def _post_process_datasets(self, datasets: List) -> List:
        return datasets

    def main(self):
        train_dataset, val_dataset = self._prepare_dataset()
        train_data_dir = os.path.join(self.args.output_dir, 'train')
        val_data_dir = os.path.join(self.args.output_dir, 'val')
        train_dataset.save_to_disk(train_data_dir)
        if val_dataset is not None:
            val_dataset.save_to_disk(val_data_dir)
        logger.info(f'cached_dataset: `{train_data_dir}`')
        if val_dataset is not None:
            logger.info(f'cached_val_dataset: `{val_data_dir}`')


def export_cached_dataset(args: Optional[Union[List[str], ExportArguments]] = None):
    return ExportCachedDataset(args).main()
