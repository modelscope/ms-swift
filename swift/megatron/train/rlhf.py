# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Optional, Union

from swift.llm.train.kto import prepare_kto_dataset
from swift.trainers.rlhf_trainer.utils import identity_data_collator
from swift.utils import get_current_device, get_logger, is_last_rank
from ..argument import MegatronRLHFArguments
from ..trainers import MegatronDPOTrainer, MegatronGRPOTrainer, MegatronKTOTrainer, MegatronRewardTrainer
from .sft import MegatronSft

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def prepare_trainer(self):
        args = self.args
        trainer_mapping = {
            'dpo': MegatronDPOTrainer,
            'grpo': MegatronGRPOTrainer,
            'kto': MegatronKTOTrainer,
            'rm': MegatronRewardTrainer
        }
        trainer_cls = trainer_mapping.get(args.rlhf_type)
        if trainer_cls is None:
            raise ValueError(f'The current Megatron-SWIFT does not support rlhf_type: {args.rlhf_type}.')
        kwargs = {}
        if args.rlhf_type == 'grpo':
            kwargs['vllm_client'] = self._prepare_vllm_client()
        return trainer_cls(args, self.template, **kwargs)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        model_mapping = {'grpo': 'train', 'kto': 'kto'}
        self.template.set_mode(model_mapping.get(self.args.rlhf_type, 'rlhf'))

    def _get_data_collator(self):
        if self.args.rlhf_type == 'grpo':
            return identity_data_collator
        return super()._get_data_collator()

    def _get_dataset(self):
        args = self.args
        train_dataset, val_dataset = super()._get_dataset()
        if args.rlhf_type == 'kto':
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)
        return train_dataset, val_dataset

    def _prepare_vllm_client(self):
        if self.args.rlhf_type != 'grpo' or (self.args.vllm_mode != 'server'):
            return
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient
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
