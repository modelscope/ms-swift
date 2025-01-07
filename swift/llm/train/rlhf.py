# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.utils import get_logger, get_model_parameter_info
from ..argument import RLHFArguments
from .kto import prepare_kto_dataset
from .sft import SwiftSft

logger = get_logger()


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        from swift.llm.infer.utils import prepare_adapter
        args = self.args
        for key in ['ref', 'reward', 'value']:
            origin_key = key
            setattr(self, f'{key}_model', None)
            if key == 'value':
                if args.rlhf_type == 'ppo':
                    key = 'reward'
                else:
                    continue
            model_id_or_path = getattr(args, f'{key}_model')
            if model_id_or_path is None:
                continue
            model_type = getattr(args, f'{key}_model_type')
            model_revision = getattr(args, f'{key}_model_revision')
            adapters = args.adapters if key == 'ref' else args.reward_adapters
            task_type = args.task_type if origin_key == 'ref' else 'seq_cls'
            # Be aware of the unexpected behavior caused by double monkey patching.
            model = args.get_model_processor(
                model=model_id_or_path, model_type=model_type, model_revision=model_revision, task_type=task_type)[0]

            model = prepare_adapter(args, model, adapters)
            if origin_key in {'ref', 'reward'}:
                model.requires_grad_(False).eval()
            else:
                model = self.prepare_model(args, model, task_type=task_type)
                logger.info(f'value_model: {model}')
                model_parameter_info = get_model_parameter_info(model)
                self.train_msg['value_model_parameter_info'] = model_parameter_info
                logger.info(f'value_model_parameter_info: {model_parameter_info}')
            setattr(self, f'{origin_key}_model', model)

        super()._prepare_model_tokenizer()

    def _prepare_template(self) -> None:
        args = self.args
        super()._prepare_template()
        model_mapping = {'kto': 'kto', 'ppo': 'pt'}
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))

        if args.rlhf_type != 'orpo' or args.model_meta.is_multimodal:
            # Avoid padding labels during the model's forward pass in multimodal models.
            self.template.loss_scale = 'last_round'

        if args.rlhf_type == 'ppo':
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        for key in ['ref', 'reward', 'value']:
            key = f'{key}_model'
            model = getattr(self, key)
            if model:
                trainer_kwargs[key] = model
        return trainer_kwargs


def rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    return SwiftRLHF(args).main()
