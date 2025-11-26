import asyncio
import random
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import json
from tree_rollout import (DataSampleTree, DivergenceStrategyMapping, FinishedReason, SampleStatus,
                          _increment_tree_idx_depth, _repeat_list_interleave, extract_last_boxed)

from swift.llm import RequestConfig, RolloutInferRequest
from swift.llm.infer.protocol import ChatCompletionResponse, RolloutOutput
from swift.plugin import MultiTurnScheduler, multi_turns


class TreeRolloutScheduler(MultiTurnScheduler):
    """
    Base class for multi-turn tree-rollout scheduling.

    Provides default implementation for multi-turn conversation management.

    CUSTOMIZATION:
        Implement the required `step()` method and optionally override `check_finished()`
        - Uses TreeRolloutScheduler's run() method infrastructure
        - Only need to implement turn transition logic in step()
        - Optionally customize termination conditions

    Attributes:
        max_tree_width (int):
            For GRPO, it must be equal to num_generations.
        max_tree_depth (int):
            Controls the maximum number of reasoning turns for a single prompt.
        root_divergence (int):
            Number of branches generated in the first-round inference at the root node.
        max_divergence (int):
            Maximum number of branches allowed for each node.
        divergence_strategy (str):
            Strategy for selecting branch nodes; defaults to logprobs.
    """

    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        super().__init__(infer_engine, max_turns, *args, **kwargs)
        self.max_tree_width = 8
        self.max_tree_depth = max_turns | 6
        self.max_divergence = 2
        self.divergence_strategy = 'logprobs'
        self.root_divergence = 1

        self.executor = ThreadPoolExecutor(max_workers=self.max_tree_width)

    async def async_infer(self,
                          infer_requests: List[Union['RolloutInferRequest', Dict[str, Any]]],
                          request_config: 'RequestConfig',
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List['RolloutOutput']:
        # dedup_requests_by_messages
        processed_request = []
        seen = set()
        uuids = []

        for item in infer_requests:
            if isinstance(item, dict):
                req = RolloutInferRequest(**item)
            else:
                req = item

            msg_key = json.dumps(req.messages, sort_keys=True)
            uuids.append(req.uuid)

            if msg_key not in seen:
                seen.add(msg_key)
                processed_request.append(req)

        request_config.logprobs = True

        outputs = await super().async_infer(processed_request, request_config, use_tqdm=use_tqdm, **kwargs)

        assert len(outputs) == len(uuids), '[Tree Rollout] Please check the max_tree_width is equal to num_generations.'

        for idx, output in enumerate(outputs):
            output.response.id = uuids[idx]

        return outputs

    async def run(self, infer_request: Union[List[RolloutInferRequest], RolloutInferRequest],
                  request_config: 'RequestConfig', **kwargs) -> List['RolloutOutput']:
        if isinstance(infer_request, RolloutInferRequest):
            infer_request = [infer_request]
        else:
            infer_request = list(infer_request)

        request_config.logprobs = True

        finished_rollout_by_root: Dict[int, List[RolloutOutput]] = {i: [] for i in range(len(infer_request))}
        finished_samples: Dict[int, List[DataSampleTree]] = {i: [] for i in range(len(infer_request))}

        samples_to_infer = []

        for root_idx in range(len(infer_request)):
            samples_to_infer.append(
                DataSampleTree(
                    tree_idx=str(root_idx),
                    request_id=infer_request[root_idx].uuid,
                    messages=infer_request[root_idx].messages,
                    status=SampleStatus.TO_INFER))

        # first step
        next_infer_step = 1
        samples_to_infer = _repeat_list_interleave(samples_to_infer, self.root_divergence)
        samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step)

        while len(samples_to_infer) > 0:
            # resolve the error: Request id xxx already running
            vllm_inputs = [
                RolloutInferRequest(messages=sample.messages, uuid=f'{sample.request_id}-{sample.tree_idx}')
                for sample in samples_to_infer
            ]

            # Get model response
            tasks = [self.infer_engine.infer_async(request, request_config, **kwargs) for request in vllm_inputs]
            outputs: List[ChatCompletionResponse] = await asyncio.gather(*tasks)

            assert len(vllm_inputs) == len(
                outputs), f'outputs length {len(outputs)} != inputs length {len(vllm_inputs)}'

            samples_last_step = deepcopy(samples_to_infer)
            samples_to_infer = []

            for idx, (sample, output) in enumerate(zip(samples_last_step, outputs)):
                assert len(output.choices) == 1, 'vllm should only generate one output'
                self.check_finished(sample, output)

                # bind the output and request
                output.id = sample.request_id
                choice = output.choices[0]
                child_sample = deepcopy(sample)
                child_sample.extend_response(choice)

                if child_sample.status == SampleStatus.FINISHED:
                    finished_samples[child_sample.root_node].append(child_sample)
                    finished_rollout_by_root[child_sample.root_node].append(
                        RolloutOutput(
                            response=output,
                            messages=deepcopy(child_sample.messages),
                            response_token_ids=deepcopy(child_sample.all_response_ids),
                            rollout_infos={'num_turns': next_infer_step},
                        ))
                else:
                    samples_to_infer.append(child_sample)

            # if we have budget, do divergence
            if len(samples_to_infer) > 0 and self.max_divergence > 1:
                for root_idx in finished_samples.keys():
                    root_to_infer_samples = [sample for sample in samples_to_infer if sample.root_node == root_idx]
                    root_finished_samples = finished_samples[root_idx]

                    budget = self.max_tree_width - len(root_finished_samples) - len(root_to_infer_samples)

                    if budget > 0 and len(root_to_infer_samples) > 0:
                        divergence_executor = DivergenceStrategyMapping[self.divergence_strategy]
                        if not divergence_executor:
                            raise ValueError(
                                f"[Tree Rollout] The divergence strategy: {self.divergence_strategy} doesn't exist.")

                        divergence_samples = divergence_executor.apply(root_idx, root_to_infer_samples, budget,
                                                                       self.max_divergence - 1)
                        samples_to_infer.extend(divergence_samples)

            # before end loop, if finished_count < max_tree_width, rollback
            if len(samples_to_infer) == 0 and any(count < self.max_tree_width
                                                  for count in [len(value) for value in finished_samples.values()]):
                samples_to_infer = self.roll_back_to_divergence(finished_samples)

            # tools call etc
            futures = [self.executor.submit(self.step, sample) for sample in samples_to_infer]
            wait(futures, return_when=ALL_COMPLETED)

            next_infer_step += 1
            samples_to_infer = _increment_tree_idx_depth(samples_to_infer, next_infer_step)

        # flatten finished outputs
        return [traj for lst in finished_rollout_by_root.values() for traj in lst]

    def step(self, sample: DataSampleTree, **kwargs):
        """
        You need to rewrite or modify this method to customize the next round of prompts, such as tools call.
        """

        # Special handling has already been done in the rollback.
        if sample.status == SampleStatus.ROLLBACK:
            sample.status = SampleStatus.TO_INFER
            return
        elif sample.status == SampleStatus.FINISH_NEXT_INFER:
            prompt = 'In this round of responses, you must generate an answer.'
        else:
            prompt = 'The answer is not correct, It seems You made a mistake, you need to recheck very carefully.'

        sample.messages.append({'role': 'user', 'content': prompt})

    def check_finished(self, sample: DataSampleTree, output: ChatCompletionResponse, **kwargs) -> bool:
        """
        Rewrite this method to add custom check logic
        """

        boxed_answer = extract_last_boxed(output.choices[0].message.content)

        if boxed_answer is not None:
            sample.status = SampleStatus.FINISHED
            sample.finished_reason = FinishedReason.ANSWER

        elif sample.status == SampleStatus.FINISH_NEXT_INFER:
            sample.status = SampleStatus.FINISHED
            sample.finished_reason = FinishedReason.MAX_INFER_STEP

        elif sample.depth >= self.max_tree_depth - 1:
            sample.status = SampleStatus.FINISH_NEXT_INFER

        return sample.status == SampleStatus.FINISHED

    def roll_back_to_divergence(
        self,
        finished_samples: Dict[int, List[DataSampleTree]],
    ) -> List[DataSampleTree]:
        """
        All nodes have completed inference, but there is still budget available, rollback.
        """

        sample_to_infer = []
        for root_idx, sample_list in finished_samples.items():
            if len(sample_list) >= self.max_tree_width:
                continue

            diff_count = self.max_tree_width - len(sample_list)
            result = random.sample(sample_list, min(diff_count, len(sample_list)))

            result_copy = deepcopy(result)

            # Randomly rollback several inference iterations; The rollback strategy can be optimized subsequently.
            for sample in result_copy:
                sample.status = SampleStatus.ROLLBACK
                truncate_len = sample.response_num
                sample.response_truncate(random.randint(1, truncate_len))

            sample_to_infer.extend(result_copy)

        return sample_to_infer


multi_turns['tree_rollout_scheduler'] = TreeRolloutScheduler
