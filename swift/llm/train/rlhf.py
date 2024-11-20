from typing import Any, Dict, List, Union

from swift.utils import get_dist_setting
from ..argument import RLHFArguments
from ..dataset import get_kl_dataset
from .sft import SwiftSft


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        args = self.args
        self.ref_model = None
        if args.ref_model:
            # Be aware of the unexpected behavior caused by double monkey patching.
            self.ref_model, _ = self._get_model_tokenizer(args.ref_model, args.ref_model_type, args.ref_model_revision)
            self.ref_model.requires_grad_(False).eval()

        super()._prepare_model_tokenizer()

    def _prepare_train(self):
        args = self.args
        mode = 'kto' if args.rlhf_type == 'kto' else 'rlhf'
        self.template.set_mode(mode)

        if args.rlhf_type != 'orpo' or args.model_meta.is_multimodal:
            # Avoid padding labels during the model's forward pass in multimodal models.
            self.template.loss_scale = 'last_round'

    def _register_post_encode_hook(self):
        models = [self.model]
        if self.ref_model:
            models.append(self.ref_model)
        self.template.register_post_encode_hook(models)

    def _prepare_kto_dataset(self, train_dataset, val_dataset):
        args = self.args
        world_size = get_dist_setting()[2]
        total_batch_size = (world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)
        if total_batch_size <= 1:
            raise ValueError('Batch size is 1 (too small). KTO will not work properly because the KL term '
                             'will be equivalent to the implied reward.')
        train_dataset = get_kl_dataset(train_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)
        val_dataset = get_kl_dataset(val_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)
        return train_dataset, val_dataset

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = self._prepare_kto_dataset(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        if self.ref_model:
            trainer_kwargs['ref_model'] = self.ref_model
        return trainer_kwargs


def rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    return SwiftRLHF(args).main()
