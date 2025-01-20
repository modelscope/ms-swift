# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import deepcopy

import json
import numpy as np

from swift.experimental.sampling.base import Sampler
from swift.llm import RequestConfig
from swift.llm.template.template_inputs import InferRequest
from swift.utils import get_logger
from .utils import get_messages_md5, get_reward

logger = get_logger()


class VanillaSampler(Sampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.sampler_engine == 'pt':
            from swift.llm import PtEngine
            _Engine = PtEngine
        elif self.args.sampler_engine == 'vllm':
            from swift.llm import VllmEngine
            _Engine = VllmEngine
        elif self.args.sampler_engine == 'lmdeploy':
            from swift.llm import LmdeployEngine
            _Engine = LmdeployEngine
        elif self.args.sampler_engine == 'no':
            _Engine = None
        else:
            raise ValueError(f'Cannot find engine name: {self.args.sampler_engine}')
        self.infer_engine = None
        if _Engine:
            self.infer_engine = _Engine(self.args.model, model_type=self.args.model_type, **self.args.engine_kwargs)
        self.caches = self.read_cache()

    def read_cache(self):
        cache_files = self.args.cache_files
        caches = {}
        for file in cache_files:
            if not os.path.exists(file):
                logger.warning(f'Cache file does not exist: {file}')
                continue
            with open(file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    content = json.loads(line)
                    uuid = content['id']
                    messages = content['messages']
                    if uuid not in caches:
                        caches[uuid] = {'choices': []}
                    assert messages[-1]['role'] == 'assistant'
                    caches[uuid]['choices'].append(messages[-1]['content'])
        return caches

    def generate(self, data):
        resp_all = []
        infer_requests = []
        sent = 0
        for _messages in data['messages']:
            messages = deepcopy(_messages)
            uuid = get_messages_md5(messages)
            if uuid in self.caches:
                choices = self.caches[uuid]['choices']
                if len(choices) == self.args.num_return_sequences:
                    continue
            if self.args.system:
                if messages[0]['role'] == 'system':
                    messages[0]['content'] = self.args.system
                else:
                    messages.insert(0, {'role': 'system', 'content': self.args.system})
            if messages[-1]['role'] == 'assistant':
                messages = messages[:-1]

            infer_request = InferRequest(messages=messages)
            for i in range(self.args.num_return_sequences):
                infer_requests.append(deepcopy(infer_request))
            sent += 1

        request_config = RequestConfig(
            max_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
        )

        resp_list = []
        if len(infer_requests) > 0:
            resp_list = self.infer_engine.infer(infer_requests, request_config=request_config)

        _cur = 0
        for _messages in data['messages']:
            uuid = get_messages_md5(_messages)
            if uuid in self.caches:
                choices = self.caches[uuid]['choices']
                if len(choices) == self.args.num_return_sequences:
                    resps = {'messages': _messages, 'choices': choices}
                    resp_all.append(resps)
                    continue

            resps = {'messages': _messages, 'choices': []}
            for j in range(self.args.num_return_sequences * _cur, self.args.num_return_sequences * (_cur + 1)):
                resps['choices'].append(resp_list[j].choices[0].message.content)
            resp_all.append(resps)
            _cur += 1
        return resp_all

    def do_sample(self, data):
        generated = []
        resp_all = self.generate(data)
        for i, resps in enumerate(resp_all):
            choices = resps['choices']
            messages = deepcopy(resps['messages'])
            uuid = get_messages_md5(messages)
            assert messages[-1]['role'] == 'assistant'
            ground_truth = messages[-1]['content']

            infer_requests = []
            for decoded in choices:
                _messages = deepcopy(messages)
                _messages[-1]['content'] = decoded
                infer_requests.append(InferRequest(messages=_messages))
            _messages = deepcopy(messages)
            _messages[-1]['content'] = ground_truth
            infer_requests.append(InferRequest(messages=_messages))
            if self.orm_model is not None:
                orm_score, _orm_mask = get_reward(
                    self.orm_model, infer_requests, ground_truths=[ground_truth] * len(infer_requests), threshold=0.0)
            else:
                orm_score = np.array([1.0] * len(infer_requests))
                _orm_mask = np.array([True] * len(infer_requests))
            if self.prm_model is not None:
                prm_score, _prm_mask = get_reward(
                    self.prm_model,
                    infer_requests,
                    ground_truths=[ground_truth] * len(infer_requests),
                    threshold=self.args.prm_threshold)
            else:
                prm_score = np.array([1.0] * len(infer_requests))
                _prm_mask = np.array([True] * len(infer_requests))

            _mask = _orm_mask & _prm_mask
            if not any(_mask):
                # Should not happen
                raise

            choices.append(ground_truth)
            choices = np.array(choices)

            if self.orm_model is None and self.prm_model is None:
                positives = choices[:-1]
                for positive in positives:
                    messages = deepcopy(messages)
                    messages[-1]['content'] = str(positive)
                    generated.append(json.dumps({'id': uuid, 'messages': messages}, ensure_ascii=False) + '\n')
            else:
                score = np.array(prm_score) + np.array(orm_score * 10)
                sorted_indices = np.argsort(score)[::-1]
                pos_indexes = sorted_indices[0:self.args.n_best_to_keep]
                pos_indexes = [i for i in pos_indexes if _mask[i]]
                neg_index = sorted_indices[-1]
                logger.info(
                    f'orm:{orm_score}, prm:{prm_score}, positive index: {pos_indexes}, negative index: {neg_index}')
                if self.args.easy_query_threshold is not None and sum([score > 0 for score in orm_score]) - 1 >= int(
                        self.args.num_return_sequences * self.args.easy_query_threshold):
                    continue
                if len(pos_indexes) > 0:
                    positives = choices[pos_indexes]
                    negative = choices[neg_index]
                    for positive in positives:
                        messages = deepcopy(messages)
                        messages[-1]['content'] = str(positive)
                        generated.append(
                            json.dumps({
                                'id': uuid,
                                'messages': messages,
                                'rejected_response': str(negative)
                            },
                                       ensure_ascii=False) + '\n')
        return generated
