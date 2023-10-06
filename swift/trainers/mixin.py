# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
import os
import shutil
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import safetensors
import torch
from datasets import Dataset as HfDataset
from peft import PeftModel
from requests.exceptions import HTTPError
from torch.nn import Module
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ADAPTER_CONFIG_NAME  # noqa
from transformers.trainer import (ADAPTER_SAFE_WEIGHTS_NAME,
                                  ADAPTER_WEIGHTS_NAME, CONFIG_NAME,
                                  PREFIX_CHECKPOINT_DIR, SAFE_WEIGHTS_NAME,
                                  TRAINER_STATE_NAME, TRAINING_ARGS_NAME,
                                  WEIGHTS_NAME, IntervalStrategy,
                                  is_peft_available)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, HubStrategy
from transformers.training_args import TrainingArguments

from swift.hub import HubApi, ModelScopeConfig, Repository
from swift.hub.check_model import check_local_model_is_latest
from swift.hub.constants import ModelVisibility
from swift.tuners import SwiftModel
from swift.utils.constants import Invoke
from swift.utils.logger import get_logger
from .utils import (can_return_loss, find_labels, get_function,
                    is_instance_of_ms_model)

logger = get_logger()


def _push_to_hub(self: Repository,
                 commit_message: str = 'Commit files to Modelscope Hub',
                 **kwargs):
    blocking = kwargs.get('blocking', True)
    self.push(commit_message)
    if not blocking:
        # Compatible with transformers
        return None, None
    else:
        return None


class PushToMsHubMixin:
    repo: Repository

    def _add_patterns_to_gitignores(
            self,
            patterns: List[str],
            commit_message: Optional[str] = None) -> None:
        # Make sure we only do this on the main process
        if not self.is_world_process_zero():
            return
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to .gitignore'

        # Get current .gitignore content
        repo_dir = self.repo.model_dir
        gitignore_path = os.path.join(repo_dir, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                current_content = f.read()
        else:
            current_content = ''

        # Add the patterns to .gitignore
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if content == '' or content.endswith('\n'):
                    content += f'{pattern}\n'
                else:
                    content += f'\n{pattern}\n'

        # Write the .gitignore file if it has changed
        if content != current_content:
            with open(gitignore_path, 'w') as f:
                logger.debug(f'Writing .gitignore file. Content: {content}')
                f.write(content)
        self.repo.push(commit_message)

    def init_hf_repo(self) -> None:
        """init ms repo. Compatible with transformers>=v4.34"""
        self.init_git_repo()

    def init_git_repo(self, at_init: bool = False) -> None:
        if not self.is_world_process_zero():
            return
        # Make sure the repo exists.
        api = HubApi()
        hub_token = self.args.hub_token
        if hub_token is None:
            hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if hub_token is not None:
            api.login(hub_token)

        hub_model_id = self.args.hub_model_id
        assert hub_model_id is not None, 'Please enter a valid hub_model_id'
        if '/' not in hub_model_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
            hub_model_id = f'{user_name}/{hub_model_id}'
            logger.info(
                f"'/' not in hub_model_id, setting hub_model_id: {hub_model_id}"
            )
            self.args.hub_model_id = hub_model_id

        visibility = ModelVisibility.PRIVATE if self.args.hub_private_repo else ModelVisibility.PUBLIC
        try:
            api.create_model(hub_model_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass

        if (os.path.exists(self.args.output_dir)
                and os.listdir(self.args.output_dir)
                and self.args.overwrite_output_dir and at_init):
            # directory not empty.
            shutil.rmtree(self.args.output_dir)
        self.repo = Repository(self.args.output_dir, hub_model_id)
        self.repo.push_to_hub = MethodType(_push_to_hub, self.repo)
        self.repo.local_dir = self.repo.model_dir  # hf compatibility

        # By default, ignore the checkpoint folders
        _commit_message = 'Add `{}` patterns to .gitignore'
        if not os.path.exists(
                os.path.join(self.args.output_dir, '.gitignore')
        ) and self.args.push_hub_strategy != 'all_checkpoints':
            self._add_patterns_to_gitignores(
                ['checkpoint-*/'], _commit_message.format('checkpoint-*/'))

        # Add 'runs/' to .gitignore, ignore tensorboard files
        self._add_patterns_to_gitignores(['runs/'],
                                         _commit_message.format('runs/'))

        # Add '*.sagemaker' to .gitignore if using SageMaker
        if os.environ.get('SM_TRAINING_ENV'):
            self._add_patterns_to_gitignores(
                ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                _commit_message.format('*.sagemaker'))

        self.push_in_progress = None

    def push_to_hub(self,
                    commit_message: str = 'End of training',
                    **kwargs) -> None:
        # user calls manually `push_to_hub` with `self.args.push_to_hub = False`
        create_model_card = kwargs.pop('create_model_card', None)
        if not hasattr(self, 'repo'):
            self.init_git_repo()
        self.save_model(_internal_call=True)

        if not self.is_world_process_zero():
            return

        self.repo.push_to_hub(commit_message, **kwargs)
        # push separately the model card to be independant from the rest of the model
        readme_path = os.path.join(self.args.output_dir, 'README.md')
        if create_model_card is None:
            create_model_card = not os.path.exists(readme_path)
        if create_model_card and self.args.should_save:
            model_name = kwargs.pop('model_name', None)
            if model_name is None and self.args.should_save:
                if self.args.hub_model_id is not None:
                    model_name = self.args.hub_model_id.split('/')[-1]
                else:
                    model_name = os.path.basename(self.args.output_dir)
            self.create_model_card(model_name=model_name, **kwargs)
            self.repo.push_to_hub('update model card README.md', **kwargs)

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        """Compatible with transformers>=4.32"""
        # Only push from one node.
        if not self.is_world_process_zero(
        ) or self.args.push_hub_strategy == 'end':
            return
        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        if is_peft_available():
            modeling_files.extend([
                ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME,
                ADAPTER_SAFE_WEIGHTS_NAME
            ])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(
                    os.path.join(checkpoint_folder, modeling_file),
                    os.path.join(output_dir, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        try:
            if self.args.push_hub_strategy == 'checkpoint':
                # Temporarily move the checkpoint just saved for the push
                tmp_checkpoint = os.path.join(output_dir, 'last-checkpoint')
                # We have to remove the "last-checkpoint" dir if it exists, otherwise the checkpoint is moved as a
                # subfolder.
                if os.path.isdir(tmp_checkpoint):
                    shutil.rmtree(tmp_checkpoint)
                shutil.move(checkpoint_folder, tmp_checkpoint)

            if self.args.save_strategy == IntervalStrategy.STEPS:
                commit_message = f'Training in progress, step {self.state.global_step}'
            else:
                commit_message = f'Training in progress, epoch {int(self.state.epoch)}'
            if self.args.push_hub_strategy == 'push_best':
                if checkpoint_folder == self.state.best_model_checkpoint:
                    self.repo.push_to_hub(
                        commit_message=commit_message,
                        blocking=False,
                        auto_lfs_prune=True)
            else:
                self.repo.push_to_hub(
                    commit_message=commit_message,
                    blocking=False,
                    auto_lfs_prune=True)
        except Exception as e:
            logger.error(f'Error when pushing to hub: {e}')
        finally:
            if self.args.push_hub_strategy == 'checkpoint':
                # Move back the checkpoint to its place
                shutil.move(tmp_checkpoint, checkpoint_folder)


class SwiftMixin:

    def __init__(self,
                 model: Union[PreTrainedModel, Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[HfDataset] = None,
                 eval_dataset: Optional[Union[HfDataset,
                                              Dict[str, HfDataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction],
                                                    Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer,
                                   torch.optim.lr_scheduler.LambdaLR] = (None,
                                                                         None),
                 preprocess_logits_for_metrics: Optional[Callable[
                     [torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 **kwargs) -> None:
        check_model = kwargs.get('check_model', True)
        if check_model and hasattr(model, 'model_dir'):
            check_local_model_is_latest(
                model.model_dir,
                user_agent={
                    Invoke.KEY:
                    Invoke.LOCAL_TRAINER,
                    Invoke.THIRD_PARTY:
                    kwargs.get(Invoke.THIRD_PARTY, Invoke.SWIFT),
                })
        # mro
        super().__init__(model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics,
                         callbacks, optimizers, preprocess_logits_for_metrics)

        if get_function(model.__class__.forward) is not get_function(
                model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)

    @staticmethod
    def _create_configuration_file(model: Module, output_dir: str) -> None:
        cfg = getattr(model, 'cfg', {})
        configuration_path = os.path.join(output_dir, 'configuration.json')
        if os.path.exists(configuration_path):
            with open(configuration_path, 'r') as f:
                res = json.load(f)
        else:
            res = {}
        if 'framework' not in res:
            res['framework'] = cfg.get('framework', 'pytorch')
        if 'task' not in res:
            res['task'] = cfg.get('task', 'text-generation')
        with open(configuration_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Compatible with swift and peft"""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving model checkpoint to {output_dir}')
        # configuration.json
        if is_instance_of_ms_model(self.model):
            model_dir = getattr(self.model, 'model_dir', None)
            if model_dir is not None:
                src_path = os.path.join(model_dir, 'configuration.json')
                dst_path = os.path.join(output_dir, 'configuration.json')
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
        else:
            self._create_configuration_file(self.model, output_dir)

        supported_classes = (SwiftModel, PreTrainedModel, PeftModel)
        # model
        save_safetensors = getattr(self.args, 'save_safetensors', False)
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            _unwrap_model = unwrap_model(self.model)
            if isinstance(_unwrap_model, supported_classes):
                _unwrap_model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=save_safetensors)
            else:
                logger.info(
                    'Trainer.model is not a `PreTrainedModel`, only saving its state dict.'
                )
                if save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, 'model.safetensors'))
                else:
                    torch.save(state_dict,
                               os.path.join(output_dir, 'pytorch_model.bin'))
        elif is_instance_of_ms_model(self.model):
            PreTrainedModel.save_pretrained(
                self.model,
                output_dir,
                state_dict=state_dict,
                safe_serialization=save_safetensors)
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=save_safetensors)
        # tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # training_args.bin
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

    def _save_checkpoint(self, model, trial, metrics=None):
        only_save_model = self.args.only_save_model
        if only_save_model:
            return self._only_save_model(model, trial, metrics)
        else:
            return super()._save_checkpoint(model, trial, metrics)

    def _only_save_model(self, model, trial, metrics=None):
        # Save model checkpoint
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.model_wrapped.save_checkpoint(output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(
                os.path.join(output_dir, TRAINER_STATE_NAME))

        # push to hub
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        train_sampler_random = self.args.train_sampler_random
        if train_sampler_random:
            return super()._get_train_sampler()
        else:
            return self._get_eval_sampler(self.train_dataset)
