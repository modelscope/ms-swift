from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel, trainer
from trl import ORPOTrainer as HFORPOTrainer

from swift.llm.utils.template import Context, Template
from swift.llm.utils.utils import sort_by_max_length
from swift.utils import get_logger
from .dpo_trainers import DPOTrainer
from .mixin import PushToMsHubMixin, SwiftMixin

logger = get_logger()


class ORPOTrainer(PushToMsHubMixin, SwiftMixin, HFORPOTrainer):

    def __init__(self,
                 *args,
                 template: Template,
                 test_oom_error=False,
                 **kwargs):
        self.template = template
        super().__init__(*args, **kwargs)
        train_ds_info = self.stat_dataset(self.train_dataset)
        val_ds_info = self.stat_dataset(self.eval_dataset)
        self.dataset_info = {
            'train_dataset': train_ds_info,
            'val_dataset': val_ds_info
        }
        if test_oom_error:
            self.train_dataset = sort_by_max_length(self.train_dataset, 20000)
        # performance
        self.perf: Dict[str, Any] = {
            'gen_time':
            0.,
            'gen_len':
            0,
            'memory': {},
            'model':
            self.model.get_trainable_parameters() if hasattr(
                self.model, 'get_trainable_parameters') else None,
        }

    def train(self, *args, **kwargs) -> torch.Tensor:
        res = super().train(*args, **kwargs)
        for i in range(torch.cuda.device_count()):
            self.perf['memory'][
                f'cuda:{i}'] = f'{torch.cuda.max_memory_reserved(i)/1024/1024/1024:.2f}GiB'
        return res

    def concat_template(self, feature):
        query: Optional[str] = feature.get('query', None)
        system: Optional[str] = feature.get('system', None)
        history: List = feature.get('history', [])
        if system is None:
            if self.template.use_default_system:
                system = self.template.default_system
        else:
            assert self.template.prefix_has_system is not None, 'not support `system`'
        res_context_list: List[Context] = []
        compute_loss_idx: List[float] = []
        if system is None:
            assert self.template.prefix != self.template.prefix_has_system, f'template.prefix: {self.template.prefix}'
            prefix = self.template.prefix
        else:
            prefix = self.template.prefix_has_system
        self.template._concat_context_list(
            prefix, res_context_list, compute_loss_idx, system=system)
        for i, (q, r) in enumerate(history):
            self.template._concat_context_list(
                [
                    *self.template.prompt,
                    '{{RESPONSE}}',
                    *self.template.chat_sep  # noqa
                ],
                res_context_list,
                compute_loss_idx,
                query=q,
                response=r,
                round0=i)  # noqa
        self.template._concat_context_list(
            self.template.prompt,
            res_context_list,
            compute_loss_idx,
            query=query,
            round0=len(history))
        res_context_list, compute_loss_idx = self.template._simplify_context_list(
            res_context_list, compute_loss_idx)

        return res_context_list, feature['response'], feature[
            'rejected_response'], compute_loss_idx

    @staticmethod
    def stat_dataset(llm_dataset) -> Any:
        _token_len = []
        from datasets import Dataset as HfDataset
        from swift.utils.np_utils import stat_array
        if isinstance(llm_dataset, HfDataset):
            chosen = llm_dataset['chosen_input_ids']
            rejected = llm_dataset['rejected_input_ids']
            for cc, rr in zip(chosen, rejected):
                _token_len.append(max(len(cc), len(rr)))
        else:
            for d in llm_dataset:
                _token_len.append(
                    max(
                        len(d['chosen_input_ids']),
                        len(d['rejected_input_ids'])))
        _, stat_str = stat_array(_token_len)
        logger.info(f'Dataset Token Length: {stat_str}')
        return stat_str
