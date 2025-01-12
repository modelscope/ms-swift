# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from copy import deepcopy
from typing import List, Union

import numpy as np

from swift.llm import RequestConfig, SwiftPipeline, load_dataset
from swift.llm.template.template_inputs import InferRequest
from swift.utils import get_logger
from .sampling_args import SamplingArguments
from ...plugin.orm import orms
from ...plugin.prm import prms

logger = get_logger()


def _get_reward(model, infer_requests: List[InferRequest], request_config=None, threshold: float = None):
    resp_list = model.infer(infer_requests, request_config=request_config)
    arr = [float(resp_list[i].choices[0].message.content) for i in range(len(resp_list))]
    logger.info(arr)

    _mask = np.array([True] * len(arr))
    if threshold is not None:
        _mask = np.array([a >= threshold for a in arr])

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

    return normalize(arr), _mask


class SwiftSampling(SwiftPipeline):
    args_class = SamplingArguments
    args: args_class

    def __init__(self, args: Union[List[str], SamplingArguments, None] = None) -> None:
        super().__init__(args)
        self.args.save_args()
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.cur_piece = 0
        self.total_piece = 1

        if self.args.data_range:
            self.cur_piece, self.total_piece = self.args.data_range

        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_rm()

        if self.args.sampler_engine == 'pt':
            from swift.llm import PtEngine
            _Engine = PtEngine
        elif self.args.sampler_engine == 'vllm':
            from swift.llm import VllmEngine
            _Engine = VllmEngine
        elif self.args.sampler_engine == 'lmdeploy':
            from swift.llm import LmdeployEngine
            _Engine = LmdeployEngine
        else:
            raise ValueError(f'Cannot find engine name: {self.args.sampler_engine}')
        self.infer_engine = _Engine(self.args.model, model_type=self.args.model_type, **self.args.engine_kwargs)

    def _prepare_model_tokenizer(self):
        args = self.args
        _, self.processor = args.get_model_processor(load_model=False)

    def _prepare_rm(self):
        if self.args.prm_model is None:
            self.prm_model = None
            logger.warning(f'prm_model is None.')
            return
        if self.args.prm_model in prms:
            self.prm_model = prms[self.args.prm_model]()
        else:
            from swift.llm import PtEngine
            self.prm_model = PtEngine(self.args.prm_model, max_batch_size=64)

        if self.args.orm_model is None:
            self.orm_model = None
            logger.warning(f'orm_model is None.')
            return
        elif self.args.orm_model in orms:
            self.orm_model = orms[self.args.orm_model]()
        else:
            from swift.llm import PtEngine
            self.orm_model = PtEngine(self.args.orm_model, max_batch_size=64)

    def _prepare_template(self) -> None:
        template = self.args.get_template(self.processor)
        self.template = template
        self.template.set_mode('train')

    def _get_dataset(self):
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        sampling_dataset, _ = load_dataset(
            args.dataset, split_dataset_ratio=0., **dataset_kwargs)
        logger.info(f'Sampling_dataset: {sampling_dataset}')
        dataset_len = len(sampling_dataset)
        piece_len = dataset_len // self.total_piece
        sampling_dataset = sampling_dataset.select(range(piece_len * self.cur_piece, piece_len * (self.cur_piece + 1)))
        return sampling_dataset

    def do_sample(self, data):
        infer_requests = []
        for _messages in data['messages']:
            messages = deepcopy(_messages)
            if self.args.system:
                if messages[0]['role'] == 'system':
                    messages[0]['content'] = self.args.system
                else:
                    messages.insert(0, {'role': 'system', 'content': self.args.system})
            if messages[-1]['role'] == 'assistant' and messages[-1]['content'] is None:
                messages = messages[:-1]
            infer_request = InferRequest(messages=messages)
            for i in range(self.args.num_return_sequences):
                infer_requests.append(deepcopy(infer_request))

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
        )
        generated = []

        resp_list = self.infer_engine.infer(infer_requests, request_config=request_config)
        batch_decoded_all = []
        for i in range(0, len(resp_list), self.args.num_return_sequences):
            batch_decoded = []
            for j in range(i, i + self.args.num_return_sequences):
                batch_decoded.append(resp_list[j].choices[0].message.content)
            batch_decoded_all.append(batch_decoded)

        for i, batch_decoded in enumerate(batch_decoded_all):
            messages = deepcopy(data['messages'][i])
            if messages[-1]['role'] != 'assistant':
                messages.append({'role': 'assistant', 'content': None})

            infer_requests = []
            for decoded in batch_decoded:
                _messages = deepcopy(messages)
                _messages[-1]['content'] = decoded
                infer_requests.append(InferRequest(messages=_messages,
                                                   ground_truths=data['ground_truth'][i]))
            _messages = deepcopy(messages)
            _messages[-1]['content'] = data['ground_truth'][i]
            infer_requests.append(InferRequest(messages=_messages,
                                               ground_truths=data['ground_truth'][i]))
            orm_score, _ = _get_reward(self.orm_model, infer_requests)
            prm_score, _mask = _get_reward(self.prm_model, infer_requests, threshold=self.args.prm_threshold)

            if not any([score > 0 for score in orm_score]):
                # Should not happen
                raise

            score = np.array(prm_score) + np.array(orm_score * 10)
            sorted_indices = np.argsort(score)[::-1]
            neg_index = sorted_indices[-1]
            pos_indexes = sorted_indices[0:self.args.n_best_to_keep]
            pos_indexes = [i for i in pos_indexes if _mask[i]]
            batch_decoded.append(data['ground_truth'][i])
            batch_decoded = np.array(batch_decoded)
            logger.info(
                f'orm:{orm_score}, prm:{prm_score}, positive index: {pos_indexes}, negative index: {neg_index}')
            if self.args.easy_query_threshold is not None and sum([score > 0 for score in orm_score]) - 1 >= int(
                    self.args.num_return_sequences * self.args.easy_query_threshold):
                continue
            if len(pos_indexes) > 0:
                positives = batch_decoded[pos_indexes]
                negative = batch_decoded[neg_index]
                for positive in positives:
                    messages = deepcopy(messages)
                    messages[-1]['content'] = str(positive)
                    generated.append(json.dumps({'messages': messages, 'rejected_response': str(negative)}) + '\n')
        return generated

    def run_sampling(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        iter_file = os.path.join(self.args.output_dir,
                                 self.args.file_prefix + f'_sampling.jsonl')
        if os.path.exists(iter_file) and not self.args.override_exist_file:
            return
        self.template.set_mode('train')
        dataset = self._get_dataset()
        dumped_ds = []
        dataset_len = len(dataset)
        total_iters = int(dataset_len // self.args.num_sampling_per_gpu_batch_size)
        if self.args.num_sampling_per_gpu_batches is None or self.args.num_sampling_per_gpu_batches > total_iters:
            self.args.num_sampling_per_gpu_batches = total_iters

        for _index in range(self.args.num_sampling_per_gpu_batches):
            logger.info(f' Sampling index:{_index}')
            generated = self.do_sample(dataset[self.args.num_sampling_per_gpu_batch_size * _index:
                                               self.args.num_sampling_per_gpu_batch_size * (_index + 1)])
            dumped_ds.extend(generated)

        with open(iter_file, 'w') as f:
            f.writelines(dumped_ds)

    def run(self):
        self.run_sampling()


def sampling_main(args: Union[List[str], SamplingArguments, None] = None):
    return SwiftSampling(args).main()
