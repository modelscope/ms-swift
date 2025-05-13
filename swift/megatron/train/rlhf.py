# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

from megatron.training import get_args, get_model, training
from megatron.training.checkpointing import load_checkpoint

from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def _patch_setup_model_and_optimizer(self):
        origin_setup_model_and_optimizer = training.setup_model_and_optimizer

        def setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs):
            args = get_args()
            ref_model = get_model(model_provider_func, model_type)
            if args.ref_load is not None:
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    ref_model, None, None, load_arg='ref_load')
            args.ref_model = ref_model
            return origin_setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

        training.setup_model_and_optimizer = setup_model_and_optimizer

    def run(self):
        self._patch_setup_model_and_optimizer()
        super().run()


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()
