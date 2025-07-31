# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Optional, Union

from swift.llm import ExportArguments
from swift.llm.train import SwiftSft
from swift.utils import get_logger

logger = get_logger()


class ExportCachedDataset(SwiftSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        super(SwiftSft, self).__init__(args)
        self.train_msg = {}  # dummy
        self.processor = None
        self._prepare_template()
        self._prepare_model_tokenizer(load_model=self.template.use_model)
        self.template.init_processor(self.processor)

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
