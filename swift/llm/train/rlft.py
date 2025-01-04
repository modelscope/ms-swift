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
from trl.models.utils import unwrap_model_for_generation
from swift.llm.infer.protocol import ChatCompletionResponse, RequestConfig
from swift.llm.template.template_inputs import StdTemplateInputs, InferRequest
from swift.utils import get_logger
from .sft import SwiftSft
from .rlhf import SwiftRLHF
from .. import PtEngine
from ..argument import RLFTArguments
from ...plugin.orm import orms
from ...plugin.prm import prms
from ...plugin.sampler import samplers

logger = get_logger()


class SwiftRLFT(SwiftRLHF):
    args_class = RLFTArguments
    args: args_class

    splitter = [
        # '.', '。', '\n\n'
    ]

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
        return [resp_list[i].choices[0].message.content for i in range(len(resp_list))]

    def rollout(self, data, trainer, step):
        os.makedirs(self.args.sampler_output, exist_ok=True)
        with open(os.path.join(self.args.sampler_output, f'step_{step}.jsonl'), 'w') as f:
            with torch.no_grad():
                eos = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
                for s in self.splitter:
                    eos.extend(self.tokenizer.encode(s, add_special_tokens=False))
                generation_config = GenerationConfig(
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.step_temperature(step),
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos,
                )

                with unwrap_model_for_generation(self.model, trainer.accelerator) as unwrapped_model:
                    generated = self.sampler(unwrapped_model, data, generation_config)
                    for i, gen in enumerate(generated):
                        _data = deepcopy(data)
                        messages = _data['_messages'][i]
                        assert messages[-1]['content'] is None
                        batch_decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True,
                                                                    clean_up_tokenization_spaces=True)
                        infer_requests = []
                        for decoded in batch_decoded:
                            _messages = deepcopy(messages)
                            _messages[-1]['content'] = decoded
                            infer_requests.append(InferRequest(messages=_messages,
                                                               ground_truths=_data['ground_truth']))
                        orm_score = self._get_reward(self.orm_model, infer_requests)
                        prm_score = self._get_reward(self.prm_model, infer_requests, request_config=RequestConfig(max_tokens=3))
                        
                        score = np.array(prm_score) + np.array(orm_score)
                        sorted_indices = np.argsort(score)
                        positive = batch_decoded[sorted_indices[0]]
                        negative = batch_decoded[sorted_indices[-1]]
                        messages[-1]['content'] = positive
                        encoded = self.template.encode(
                            StdTemplateInputs.from_dict({'messages': messages, 'rejected_response': negative},
                                                        tools_prompt=self.args.tools_prompt))
                        encoded.pop('_messages', None)
                        generated.append(encoded)
                        f.write(json.dumps({'messages': messages}) + '\n')
                return generated

    def step_temperature(self, step):
        # Linear
        step_wise = (self.args.end_temperature - self.args.temperature) / self.args.training_args.max_steps
        return self.args.temperature + step_wise * step

    def step_reward_threshold(self, step):
        # Linear
        step_wise = (self.args.end_threshold - self.args.start_threshold) / self.args.training_args.max_steps
        return self.args.start_threshold + step_wise * step

    @staticmethod
    @contextlib.contextmanager
    def switch_dataset(trainer, sampled_ds):
        origin_dataset: Dataset = trainer.train_dataset
        trainer.train_dataset = sampled_ds
        yield
        origin_dataset = origin_dataset.shuffle()
        trainer.train_dataset = origin_dataset

    def train(self, trainer):
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')
        self._prepare_rm()
        self._prepare_sampler()
        new_dataset = []
        for _iter in range(self.args.num_rollout_iters):
            train_dataloader = trainer.get_train_dataloader()
            for _index, batch in enumerate(train_dataloader):
                new_data = self.rollout(batch, trainer, _iter)
                new_dataset.extend(new_data)
                if _index > self.args.num_rollout_batches:
                    break

            with SwiftRLFT.switch_dataset(new_dataset):
                trainer.train(trainer.args.resume_from_checkpoint)

        return self._save_trainer_state(trainer)


def generate(
        lm_backbone: PreTrainedModel, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Code borrowed from trl"""
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
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
        responses.append(response)
    return responses


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
