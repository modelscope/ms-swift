# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import torch.nn.functional as F
from functools import partial

from swift.utils import get_logger
from .trainer import Trainer
from .utils import gather_for_unpadded_tensors

logger = get_logger()


class EmbeddingTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        mrl_dims = self.args.mrl_dims
        if mrl_dims and self.compute_loss_func is not None:
            origin_loss_func = self.compute_loss_func

            def mrl_loss_func(outputs, labels, **kwargs):
                # Matryoshka Representation Learning: compute loss on each truncated dimension
                # and aggregate with the corresponding weights.
                last_hidden_state = outputs['last_hidden_state']
                loss = None
                for dim, weight in mrl_dims.items():
                    if dim > last_hidden_state.shape[-1]:
                        logger.warning_once(f'MRL: skipping dimension {dim} because it exceeds the model hidden size '
                                            f'({last_hidden_state.shape[-1]}).')
                        continue
                    sliced = F.normalize(last_hidden_state[..., :dim], p=2, dim=-1)
                    cur_loss = weight * origin_loss_func({'last_hidden_state': sliced}, labels, **kwargs)
                    loss = cur_loss if loss is None else loss + cur_loss
                return loss

            self.compute_loss_func = mrl_loss_func

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += [
            key for key in ['_dataset_group', '_infonce_config'] if key not in self._signature_columns
        ]

    @staticmethod
    def _uses_dataset_batching(dataset):
        dataset = getattr(dataset, 'dataset', dataset)
        return '_dataset_group' in getattr(dataset, 'column_names', [])

    def get_train_dataloader(self, skip_batches=0):
        if self._uses_dataset_batching(self.train_dataset):
            if self.args.loss_type != 'infonce' or self.template.sequence_parallel_size > 1:
                raise ValueError('Dataset-specific InfoNCE requires loss_type=infonce and sequence_parallel_size=1.')
        return super().get_train_dataloader(skip_batches)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if isinstance(dataset, str):
            dataset = self.eval_dataset[dataset]
        if not self._uses_dataset_batching(dataset):
            return super().get_eval_dataloader(eval_dataset)
        from swift.dataloader import DataLoaderShard
        from swift.dataloader.dataset_sampler import DatasetBatchSamplerShard
        sampler = DatasetBatchSamplerShard(
            dataset['_dataset_group'],
            self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            data_seed=self.args.data_seed)
        if hasattr(dataset, 'strict'):
            dataset.strict = True
            dataset.n_try_fetch = 1
        params = {
            'batch_sampler': sampler,
            'collate_fn': self.data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
            'prefetch_factor': self.args.dataloader_prefetch_factor,
        }
        params.update(self._maybe_multiprocessing_context(self.args))
        return DataLoaderShard(dataset, device=self.accelerator.device, **params)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        group = inputs.pop('_dataset_group', None)
        config = inputs.pop('_infonce_config', None)
        loss_func = self.compute_loss_func
        if config is not None:
            if self.args.loss_type != 'infonce':
                raise ValueError('Dataset-specific infonce_config requires loss_type=infonce.')
            self.compute_loss_func = partial(loss_func, infonce_config=json.loads(config), dataset_group=group)
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            self.compute_loss_func = loss_func

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        self.gather_function = gather_for_unpadded_tensors
        return output
