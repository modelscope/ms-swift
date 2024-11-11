# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from swift.llm.utils import get_model_with_value_head
from swift.trainers import TrainerFactory
from swift.utils import get_logger, get_main, seed_everything
from .sft import prepare_dataset, prepare_model_template_train, trainer_train
from .utils import TEMPLATE_MAPPING, RLHFArguments

logger = get_logger()


def llm_rlhf(args: RLHFArguments) -> Dict[str, Any]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)

    is_generation = TEMPLATE_MAPPING[args.template_type].get('is_generation', False)
    if is_generation:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct.")

    kwargs = {}
    if args.rlhf_type == 'ppo':
        from copy import deepcopy
        reward_model_args, value_model_args = deepcopy(args), deepcopy(args)
        args_to_modified = ['model_id_or_path', 'model_type', 'model_revision']
        for model_args in [reward_model_args, value_model_args]:
            for arg in args_to_modified:
                setattr(model_args, arg, getattr(args, f'reward_{arg}'))
        reward_model_args.ref_model_free = True  # avoid to create ref model
        value_model_args.ref_model_free = True
        reward_model, _, _, _ = prepare_model_template_train(reward_model_args)
        reward_model.requires_grad_(False).eval()

        reward_model = get_model_with_value_head(reward_model)  # add and load value head
        # hack here to customize the value model
        value_model, _, _, _ = prepare_model_template_train(value_model_args)
        value_model = get_model_with_value_head(value_model)
        kwargs['reward_model'] = reward_model
        kwargs['value_model'] = value_model

    msg = {}
    model, ref_model, template, callbacks = prepare_model_template_train(args)

    with TrainerFactory.patch_template(args, template):
        train_dataset, val_dataset = prepare_dataset(args, template, msg)

        return trainer_train(
            args,
            model,
            template,
            train_dataset,
            val_dataset,
            callbacks=callbacks,
            msg=msg,
            ref_model=ref_model,
            **kwargs)


rlhf_main = get_main(RLHFArguments, llm_rlhf)
