
from ..dataset import EncodePreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset, load_dataset
from swift.llm import ExportArguments
from swift.llm.train import SwiftSft
from typing import Union, List


class ExportCachedDataset(SwiftSft):
    args_class = ExportArguments
    args: args_class

    def __init__(self, args: Union[List[str], ExportArguments, None] = None) -> None:
        super(SwiftSft, self).__init__(args)
        self.train_msg = {}  # dummy
        self.processor = None
        self._prepare_template()
        self._prepare_model_tokenizer(load_model=self.template.use_model)
        self.template.init_processor(self.processor)

    def _save_val_dataset(self, val_dataset):
        pass

    def run(self):
        self.args.lazy_tokenize = False
        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        return 


def export_cached_dataset(args: Union[List[str], ExportArguments, None] = None):
    return ExportCachedDataset(args).main()
