# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronRLHFArguments
from swift.pipelines.train import prepare_kto_dataset
from swift.utils import get_current_device, get_logger, is_last_rank
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        trainer_mapping = {
            'dpo': 'MegatronDPOTrainer',
            'gkd': 'MegatronGKDTrainer',
            'grpo': 'MegatronGRPOTrainer',
            'kto': 'MegatronKTOTrainer',
            'rm': 'MegatronRewardTrainer'
        }
        module = importlib.import_module('swift.megatron.trainers')
        trainer_cls_name = trainer_mapping.get(args.rlhf_type)
        if trainer_cls_name is None:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        trainer_cls = getattr(module, trainer_cls_name)
        kwargs = {}
        if args.rlhf_type in ('grpo', 'gkd'):
            kwargs['vllm_client'] = self._prepare_vllm_client()
        return trainer_cls(args, self.template, **kwargs)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        model_mapping = {'grpo': 'train', 'gkd': 'train', 'kto': 'kto'}
        self.template.set_mode(model_mapping.get(self.args.rlhf_type, 'rlhf'))

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _prepare_vllm_client(self):
        # Only prepare vLLM client for server mode
        if self.args.rlhf_type not in ('grpo', 'gkd') or self.args.vllm_mode != 'server':
            return None
        # GKD may not use vLLM (off-policy mode)
        if not getattr(self.args, 'use_vllm', False):
            return None

        from swift.rlhf_trainers.vllm_client import VLLMClient
        vllm_client = None
        if is_last_rank():
            logger.info('Start connecting to vLLM server')
            vllm_client = VLLMClient(
                base_urls=self.args.vllm_server_base_url,
                hosts=self.args.vllm_server_host,
                server_ports=self.args.vllm_server_port,
                group_ports=self.args.vllm_server_group_port,
                connection_timeout=self.args.vllm_server_timeout)
            vllm_client.close_communicator()
            vllm_client.init_communicator(device=get_current_device())
            logger.info('Connected to vLLM server')
        return vllm_client


def megatron_rlhf_main(args: Optional[Union[List[str], MegatronRLHFArguments]] = None):
    return MegatronRLHF(args).main()
