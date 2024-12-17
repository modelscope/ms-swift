# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from swift.utils import patch_getattr
from ..argument import RLHFArguments
from .kto import prepare_kto_dataset
from .sft import SwiftSft


class SwiftRLHF(SwiftSft):
    args_class = RLHFArguments
    args: args_class

    def _prepare_model_tokenizer(self):
        args = self.args
        self.ref_model = None
        if args.ref_model:
            # Be aware of the unexpected behavior caused by double monkey patching.
            self.ref_model, _ = args.get_model_processor(
                model=args.ref_model, model_type=args.ref_model_type, model_revision=args.ref_model_revision)
            self.ref_model.requires_grad_(False).eval()

        super()._prepare_model_tokenizer()

    def _prepare_template(self, use_chat_template: bool) -> None:
        args = self.args
        super()._prepare_template(use_chat_template=use_chat_template)
        mode = 'kto' if args.rlhf_type == 'kto' else 'rlhf'
        self.template.set_mode(mode)

        if args.rlhf_type != 'orpo' or args.model_meta.is_multimodal:
            # Avoid padding labels during the model's forward pass in multimodal models.
            self.template.loss_scale = 'last_round'

    @classmethod
    def prepare_model(cls, args, model, *_args, **kwargs):
        model = super().prepare_model(args, model, *_args, **kwargs)
        if args.rlhf_type == 'rm':
            from trl import AutoModelForCausalLMWithValueHead
            lm_head_namings = ['lm_head', 'embed_out']
            if not any(hasattr(model, attribute) for attribute in lm_head_namings):
                model.lm_head = None  # avoid error
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
            patch_getattr(AutoModelForCausalLMWithValueHead, 'pretrained_model')
        return model

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _get_trainer_kwargs(self):
        trainer_kwargs = {}
        if self.ref_model:
            trainer_kwargs['ref_model'] = self.ref_model
        return trainer_kwargs


def rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    return SwiftRLHF(args).main()
