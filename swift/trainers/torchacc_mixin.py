# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Optional

from transformers import PreTrainedModel, is_datasets_available

from swift.utils import use_torchacc
from swift.utils.torchacc_utils import (patch_clip_grad_norm, save_ta_ddp_checkpoint, save_ta_fsdp_checkpoint,
                                        ta_eval_dataloader, ta_load_optimizer_and_scheduler,
                                        ta_save_optimizer_and_scheduler, ta_test_dataloader, ta_train_dataloader,
                                        ta_trim_graph)


class TorchAccMixin:

    def __init__(self, *args, **kwargs):
        if use_torchacc():
            patch_clip_grad_norm(self.accelerator)
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if not use_torchacc():
            return super().get_train_dataloader()

        if is_datasets_available():
            import datasets

        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        return ta_train_dataloader(train_dataset, data_collator, self._get_train_sampler(), self.args,
                                   self._train_batch_size)

    def get_eval_dataloader(self, eval_dataset=None):

        if not use_torchacc():
            return super().get_eval_dataloader(eval_dataset)

        if is_datasets_available():
            import datasets

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError('Trainer: evaluation requires an eval_dataset.')
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description='evaluation')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='evaluation')

        return ta_eval_dataloader(eval_dataset, data_collator, self._get_eval_sampler(eval_dataset), self.args)

    def get_test_dataloader(self, test_dataset):

        if not use_torchacc():
            return super().get_test_dataloader(test_dataset)

        if is_datasets_available():
            import datasets

        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description='test')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='test')

        return ta_test_dataloader(test_dataset, data_collator, self._get_eval_sampler(test_dataset), self.args)

    def _save_tpu(self, output_dir: Optional[str] = None):

        if not use_torchacc():
            return super()._save_tpu(output_dir)

        import torch_xla.core.xla_model as xm

        # Compatible with swift and peft
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        if xm.is_master_ordinal(local=False):
            os.makedirs(output_dir, exist_ok=True)
            # configuration.json
            model_dir = getattr(self.model, 'model_dir', None)
            if model_dir is not None:
                src_path = os.path.join(model_dir, 'configuration.json')
                dst_path = os.path.join(output_dir, 'configuration.json')
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
            else:
                self._create_configuration_file(self.model, output_dir)
            self._save_sft_args(output_dir)
            # generation_config
            generation_config = getattr(self.args, 'generation_config', None)
            if generation_config is not None:
                generation_config.save_pretrained(output_dir)

        # model
        if self.args.fsdp_num > 1:
            save_ta_fsdp_checkpoint(self.model, self.tokenizer, self.args, output_dir)
        else:
            save_ta_ddp_checkpoint(self.model, self.tokenizer, self.args, output_dir)

        # additional files
        if xm.is_master_ordinal(local=False):
            if self.args is not None and self.args.sft_type == 'full':
                additional_files = getattr(self.args, 'additional_saved_files',
                                           None) or [] + ['preprocessor_config.json']
                if model_dir is not None:
                    for file in additional_files:
                        src_path = os.path.join(model_dir, file)
                        dst_path = os.path.join(output_dir, file)
                        if os.path.isfile(src_path):
                            shutil.copy(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path)

    def _load_optimizer_and_scheduler(self, checkpoint):

        if not use_torchacc() or self.args.fsdp_num == 1:
            return super()._load_optimizer_and_scheduler(checkpoint)

        self.optimizer, self.lr_scheduler = ta_load_optimizer_and_scheduler(self.optimizer, self.lr_scheduler,
                                                                            checkpoint, self.args.device)

    def _save_optimizer_and_scheduler(self, output_dir):
        if not use_torchacc() or not self.args.fsdp_num == 1:
            return super()._save_optimizer_and_scheduler(output_dir)

        return ta_save_optimizer_and_scheduler(self.optimizer, self.lr_scheduler, output_dir)

    def _maybe_log_save_evaluate(self, tr_loss, *args, **kwargs):
        if use_torchacc() and self.control.should_log:
            ta_trim_graph()
        super()._maybe_log_save_evaluate(tr_loss, *args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        if use_torchacc():
            if model is None:
                model = self.model
            # Loading checkpoint of TorchAcc has been done in tuner.py when
            # sft_type is 'full'.
            if self.args.fsdp_num > 1:
                model = model._get_underlay_model().module.module
            if isinstance(model, PreTrainedModel):
                return
        return super()._load_from_checkpoint(resume_from_checkpoint, model)
