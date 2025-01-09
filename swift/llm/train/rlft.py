# Copyright (c) Alibaba, Inc. and its affiliates.
import contextlib
import json
import os
from copy import deepcopy
from typing import List, Union, Tuple

import numpy as np
import torch
from datasets import Dataset
from modelscope import GenerationConfig
from transformers import PreTrainedModel

from swift.llm.template.template_inputs import InferRequest
from swift.utils import get_logger, get_dist_setting, get_model_parameter_info
from .rlhf import SwiftRLHF
from .. import PtEngine, RequestConfig, LmdeployEngine
from ..argument import RLFTArguments
from ...plugin.orm import orms
from ...plugin.prm import prms
from ...plugin.sampler import samplers
from ...trainers import TrainerFactory

logger = get_logger()


class SwiftRLFT(SwiftRLHF):
    args_class = RLFTArguments
    args: args_class

    def _prepare_rm(self):
        if self.args.prm_model is None:
            self.prm_model = None
            return
        if self.args.prm_model in prms:
            self.prm_model = prms[self.args.prm_model]()
        else:
            self.prm_model = PtEngine(self.args.prm_model, max_batch_size=64)

        if self.args.orm_model is None:
            self.orm_model = None
            return
        elif self.args.orm_model in orms:
            self.orm_model = orms[self.args.orm_model]()
        else:
            self.orm_model = PtEngine(self.args.orm_model, max_batch_size=64)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        self.template.set_mode('train')

    def _sample(self, model, batch, generation_config: GenerationConfig):
        queries = batch["input_ids"]
        generation_config.num_return_sequences = self.args.num_return_sequences
        generation_config.return_legacy_cache = False
        # generation_config.num_beam_groups = 5
        # generation_config.num_beams = 10
        # generation_config.do_sample = False
        # generation_config.diversity_penalty = 0.1

        with torch.no_grad():
            responses = batch_generation(model, queries,
                                         local_rollout_forward_batch_size=queries.shape[0],
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         generation_config=generation_config)
            return responses

    def _prepare_sampler(self):
        if self.args.sampler_type in samplers:
            self.sampler = samplers[self.args.sampler_type]()
        elif self.args.sampler_type == 'sample':
            self.sampler = self._sample
        elif self.args.sampler_type == 'mcts':
            pass
        else:
            raise NotImplementedError

    def _get_reward(self, model, infer_requests: List[InferRequest], request_config=None):
        resp_list = model.infer(infer_requests, request_config=request_config)
        arr = [float(resp_list[i].choices[0].message.content) for i in range(len(resp_list))]

        def normalize(arr):
            min_val = np.min(arr)
            max_val = np.max(arr)
            if min_val == max_val:
                if min_val == 0:
                    constant_value = 0.0
                else:
                    constant_value = min(1.0, min_val)
                return np.full_like(arr, fill_value=constant_value, dtype=np.float64)
            normalized = (arr - min_val) / (max_val - min_val + 1e-5)
            return normalized

        return normalize(arr)

    def rollout(self, data, trainer, step):
        infer_requests = []
        for messages in data["_messages"]:
            messages = deepcopy(messages)
            assert messages[0]['role'] == 'system'
            assert messages[-1]['role'] == 'assistant' and messages[-1]['content'] is None
            messages[0]['content'] = self.args.system
            messages = messages[:-1]
            infer_request = InferRequest(messages=messages)
            for i in range(self.args.num_return_sequences):
                infer_requests.append(infer_request)

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.step_temperature(step),
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            # n=self.args.num_return_sequences,
        )
        res = []
        origin = []

        resp_list = self.infer_engine.infer(infer_requests, request_config=request_config)
        batch_decoded_all = []
        for i in range(0, len(resp_list), self.args.num_return_sequences):
            batch_decoded = []
            for j in range(i, i+self.args.num_return_sequences):
                batch_decoded.append(resp_list[j].choices[0].message.content)
            batch_decoded_all.append(batch_decoded)

        for i, batch_decoded in enumerate(batch_decoded_all):
            _data = deepcopy(data)
            messages = _data['_messages'][i]
            assert messages[-1]['content'] is None
            infer_requests = []
            for decoded in batch_decoded:
                _messages = deepcopy(messages)
                _messages[-1]['content'] = decoded
                infer_requests.append(InferRequest(messages=_messages,
                                                   ground_truths=_data['ground_truth'][i]))
            _messages = deepcopy(messages)
            _messages[-1]['content'] = _data['ground_truth'][i]
            infer_requests.append(InferRequest(messages=_messages,
                                               ground_truths=_data['ground_truth'][i]))
            orm_score = self._get_reward(self.orm_model, infer_requests)
            prm_score = self._get_reward(self.prm_model, infer_requests)

            if not any([score > 0 for score in orm_score]):
                raise

            score = np.array(prm_score) + np.array(orm_score * 10)
            sorted_indices = np.argsort(score)
            batch_decoded.append(_data['ground_truth'][i])
            logger.info(
                f'orm:{orm_score}, prm:{prm_score}, positive index: {sorted_indices[-1]}, negative index: {sorted_indices[0]}')
            if sum([score > 0 for score in orm_score]) - 1 >= int(self.args.num_return_sequences * 0.8):
                continue
            positive = batch_decoded[sorted_indices[-1]]
            negative = batch_decoded[sorted_indices[0]]
            messages[-1]['content'] = positive
            origin.append(json.dumps({'messages': messages, 'rejected_response': negative}) + '\n')
        return origin

    def step_temperature(self, step):
        # Linear
        step_wise = (self.args.end_temperature - self.args.temperature) / self.args.num_rollout_iters
        return self.args.temperature + step_wise * step

    def step_reward_threshold(self, step):
        # Linear
        step_wise = (self.args.end_threshold - self.args.start_threshold) / self.args.num_rollout_iters
        return self.args.start_threshold + step_wise * step

    @staticmethod
    @contextlib.contextmanager
    def switch_dataset(trainer, sampled_ds):
        origin_dataset: Dataset = trainer.train_dataset
        trainer.train_dataset = sampled_ds
        yield
        origin_dataset = origin_dataset.shuffle()
        trainer.train_dataset = origin_dataset

    def _rollout_or_load(self, _iter, trainer):
        _, local_rank, world_size, _ = get_dist_setting()
        logger.info(f'Starting iter:{_iter}')
        if hasattr(trainer, 'origin_dataset'):
            trainer.train_dataset = trainer.origin_dataset.shuffle()
        iter_file = os.path.join(self.args.sampler_output, f'step_{_iter}.jsonl')
        if not os.path.exists(iter_file) or not self.args.use_cache_dataset:
            self.template.set_mode('train')
            train_dataloader = trainer.get_train_dataloader()
            dumped_ds = []

            for _index, batch in enumerate(train_dataloader):
                self.template.set_mode('rlhf' if self.args.rlft_type != 'causal_lm' else 'train')
                logger.info(f'Rolling out index:{_index}')
                origin = self.rollout(batch, trainer, _iter)
                self.template.set_mode('train')
                dumped_ds.extend(origin)
                if _index >= self.args.num_rollout_batches - 1:
                    break

            if world_size > 1:
                from accelerate.utils import gather_object
                dumped_ds = gather_object(dumped_ds)

            if local_rank <= 0:
                with open(iter_file, 'w') as f:
                    f.writelines(dumped_ds)

            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        trainer.train()
        return self._save_trainer_state(trainer)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.model, self.processor = args.get_model_processor(load_model = self.args.task != 'rollout')

        if self.args.task != 'rollout':
            if hasattr(self.model, 'hf_device_map'):
                logger.info(f'model.hf_device_map: {self.model.hf_device_map}')

            logger.info(f'model_info: {self.model.model_info}')

            if getattr(self.model, 'generation_config', None):
                self._prepare_generation_config()
            self._prepare_gradient_checkpointing()

    def run(self):
        args = self.args
        if self.args.task == 'rollout':
            self.template.set_mode('train')
        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        data_collator = self._get_data_collator()
        if self.args.task == 'rollout':
            from swift.llm import get_model_tokenizer
            self.model = get_model_tokenizer('Qwen/Qwen2.5-0.5B-Instruct')[0]
            self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        else:
            logger.info(f'model: {self.model}')
            model_parameter_info = get_model_parameter_info(self.model)
            self.train_msg['model_parameter_info'] = model_parameter_info
            logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        if self.args.task == 'rollout':
            self._prepare_rm()
            self._prepare_sampler()
            self.infer_engine = LmdeployEngine(self.args.model, model_type=self.args.model_type)
            os.makedirs(self.args.sampler_output, exist_ok=True)
            self._rollout_or_load(self.args.iter, trainer)
        else:
            return self.train(trainer)


def generate(
        lm_backbone: PreTrainedModel, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Code borrowed from trl"""
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    # input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=queries,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
    )
    return output.sequences[:, context_length:]


@torch.no_grad()
def batch_generation(
        model: torch.nn.Module,
        queries: torch.Tensor,
        local_rollout_forward_batch_size: int,
        pad_token_id: int,
        generation_config: GenerationConfig,
):
    """Code borrowed from trl"""
    responses = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i: i + local_rollout_forward_batch_size]
        response = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        response = response.reshape(local_rollout_forward_batch_size, -1, response.shape[-1])
        responses.append(response)
    return torch.cat(responses, 0)


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
