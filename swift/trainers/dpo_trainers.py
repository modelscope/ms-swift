from typing import Union, Dict

from torch import nn
from transformers import PreTrainedModel
from trl import DPOTrainer as HFDPOTrainer

from swift.llm import Template
from swift.trainers.mixin import PushToMsHubMixin, SwiftMixin
from swift.utils import get_logger

logger = get_logger()


class DPOTrainer(PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def __init__(self, *args, template: Template, **kwargs):
        self.template = template
        super().__init__(*args, **kwargs)
        self.stat_dataset(self.train_dataset)
        self.stat_dataset(self.eval_dataset)

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        return self.template.encode(feature)

    @staticmethod
    def stat_dataset(llm_dataset) -> None:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            prompt = llm_dataset['prompt']
            chosen = llm_dataset['chosen']
            rejected = llm_dataset['rejected']
            for ii, cc, rr in zip(prompt, chosen, rejected):
                _token_len.append(len(ii) + max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                _token_len.append(len(d['prompt']))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
