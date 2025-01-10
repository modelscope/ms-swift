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
                infer_requests.append(deepcopy(infer_request))

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.step_temperature(step),
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            # n=self.args.num_return_sequences,
        )
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

    def _rollout_or_load(self, _iter, trainer):
        logger.info(f'Starting iter:{_iter}')
        trainer.train_dataset = trainer.train_dataset.shuffle().select(range(self.args.gpu * self.args.per_device_train_batch_size * self.args.num_rollout_batches, (self.args.gpu+1) * self.args.per_device_train_batch_size * self.args.num_rollout_batches))
        iter_file = os.path.join(self.args.sampler_output, f'rollout_iter_{_iter}_gpu_{self.args.gpu}.jsonl')
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

            with open(iter_file, 'w') as f:
                f.writelines(dumped_ds)

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        trainer.train()
        return self._save_trainer_state(trainer)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.model, self.processor = args.get_model_processor(load_model=self.args.task != 'rollout')

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
            iter_file = os.path.join(self.args.sampler_output, f'rollout_iter_{self.args.iter}_gpu_{self.args.gpu}.jsonl')
            if os.path.exists(iter_file) and self.args.use_cache_dataset:
                return
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
            self.infer_engine = LmdeployEngine(self.args.model, model_type=self.args.model_type)
            os.makedirs(self.args.sampler_output, exist_ok=True)
            self._rollout_or_load(self.args.iter, trainer)
        else:
            return self.train(trainer)


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
