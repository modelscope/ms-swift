from copy import deepcopy
import numpy as np
import json
import time

from swift.llm import InferRequest
from swift.llm.infer.protocol import UsageInfo
from swift.utils import get_logger

from .base import Sampler
from .utils import get_reward
from .sampling_args import SamplingArguments

from typing import Union, List

logger = get_logger('./output/sampler/mcts.log')


SYS_PROMPT = """You are a super intelligent AI, you can solve any math problem step by step. 

REMEMBER: Each step should stop with a 'ки'. Final answer should start with '# Answer'. 

Here is an example:

user
Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year? 

assistant
Step 1: Janet spends 3 hours + 5 hours = <<3+5=8>>8 hours per week on music lessons. ки 
Step 2: She spends 40 * 3 = <<40*3=120>>120 on clarinet lessons per week. ки 
Step 3: She spends 28 * 5 = <<28*5=140>>140 on piano lessons per week. ки 
Step 4: Janet spends 120 + 140 = <<120+140=260>>260 on music lessons per week. ки 
Step 5: She spends 260 * 52 = <<260*52=13520>>13520 on music lessons in a year. ки 
# Answer 13520 ки 

Now answer the question:
"""
NXT_PROMPT = """Please continue.
"""

SEP_TOKEN = "ки\n"

system_message = {
    "role": "system",
    "content": SYS_PROMPT,
}
next_message = {
    "role": "user",
    "content": NXT_PROMPT,
}

def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
    if isinstance(answers, str):
        answers = [answers]
    results = []
    for answer in answers:
        results.append("# Answer" in answer)
    return results

class LanguageNode:

    def __init__(self,
                 step: str = None,
                 parent: "LanguageNode" = None,):
        self.parent = parent
        if parent:
            self.path = parent.path[:] + [step]
            self.answer = parent.answer + step + SEP_TOKEN
            self.depth = parent.depth + 1
        else:
            self.path = []
            self.answer = ""
            self.depth = 0

        self.active_children = []
        self.children = []
        self.visit_count = 0
        self.process_reward = 0.0
        self.outcome_reward = 0.0
        self.terminated = False

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def visit(self):
        self.visit_count += 1

    def init_and_update_value(self, value):
        self.outcome_reward = (self.outcome_reward * self.visit_count + value) / (self.visit_count + 1)

    def add_child(self, child: "LanguageNode"):
        self.children.append(child)
        if not child.terminated:
            self.active_children.append(child)

    def __lt__(self, other):
        return self.outcome_reward < other.outcome_reward


class MctsSampler(Sampler):

    def __init__(self, input_args: SamplingArguments):
        super().__init__(input_args)
        self.usage_info = UsageInfo(0,0,0)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.infer_kwargs = {}
        if args.sampler_engine == 'client':
            import os
            from swift.llm import InferClient
            api_key = os.getenv('DASHSCOPE_API_KEY')
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            self.infer_engine = InferClient(base_url=base_url, api_key=api_key)
            self.infer_kwargs['model'] = args.model
        else:
            _Engine = self.get_infer_engine()
            self.infer_engine = _Engine(self.args.model, model_type=self.args.model_type, **self.args.engine_kwargs)

    def get_infer_engine(self):
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
        return _Engine

    def _prepare_template(self) -> None:
        pass

    def search_single(self, query, ground_truth):
        def _UCT(node: LanguageNode):
            alpha = _args.process_reward_rate
            value = alpha * node.process_reward + (1 - alpha) * node.outcome_reward
            if node.is_root():
                return value

            exploitation_score = value
            exploration_score = (_args.exploration_rate
                                 * np.sqrt(np.log(node.parent.visit_count) / (node.visit_count + 1)))

            return exploration_score + exploitation_score

        def _select(node: LanguageNode):
            while not node.is_leaf():
                node = max(node.active_children, key=lambda x: _UCT(x))
            return node

        def _expand(node: LanguageNode):
            # s_time = time.time()
            prompt_message = {
                "role": "user",
                "content": query,
            }
            if node.is_root():
                infer_request = InferRequest([system_message, prompt_message])
            else:
                history_message = {
                    "role": "assistant",
                    "content": node.answer,
                }
                infer_request = InferRequest([system_message, prompt_message, history_message, next_message])
            expand_request_config = deepcopy(request_config)
            n = _args.num_return_sequences - len(node.children)
            while n > 0:
                expand_request_config.n = n if n <= 4 else 4
                expand_request_config.num_return_sequences = expand_request_config.n
                expand_request_config.num_beams = expand_request_config.n
                expand_request_config.seed += 1
                responses = self.infer_engine.infer(
                    [infer_request],
                    expand_request_config,
                    **self.infer_kwargs,
                )
                n -= len(responses[0].choices)
                for key, value in self.usage_info.__dict__.items():
                    update_value = getattr(responses[0].usage, key, None) + value
                    setattr(self.usage_info, key, update_value)
                for choice in responses[0].choices:
                    output = choice.message.content.rstrip(SEP_TOKEN + '\n')
                    output = output.split(SEP_TOKEN)[0]
                    child = LanguageNode(step=output, parent=node)
                    if check_terminate(child.answer)[0]:
                        child._terminated = True
                        orm_infer_requests = [InferRequest([{"role": "assistant", "content": output}])]
                        orm_score, _orm_mask = get_reward(
                            self.orm_model, orm_infer_requests, ground_truths=[ground_truth] * len(orm_infer_requests),
                            threshold=0.0)
                        child.init_and_update_value(orm_score[0])
                        if child.outcome_reward == 1:
                            terminate_correct.append(child.answer)
                        else:
                            terminate_incorrect.append(child.answer)
                    node.add_child(child)
            # logger.info(f"expand time: {time.time() - s_time}")
            # s_time = time.time()
            if self.prm_model:
                prm_infer_requests = []
                for child in node.children:
                    prm_message = {"role": "assistant", "content": child.answer}
                    prm_infer_requests.append(InferRequest([prompt_message, prm_message]))
                prm_score, _prm_mask = get_reward(
                    self.prm_model,
                    prm_infer_requests,
                    ground_truths=[ground_truth] * len(prm_infer_requests),
                    threshold=0.0)
                for child, score in zip(node.children, prm_score):
                    child.process_reward = score
            # logger.info(f"prm time: {time.time() - s_time}")

        def _rollout(node: LanguageNode):
            rollout_iter_index = 0
            prompt_message = {
                "role": "user",
                "content": query,
            }
            rollout_request_config = deepcopy(request_config)
            rollout_request_config.temperature = 0.0
            rollout_request_config.max_tokens = 500
            rollout_nodes = node.active_children[:]
            history_messages = []
            for child in rollout_nodes:
                history_message = {
                    "role": "assistant",
                    "content": child.answer,
                }
                history_messages.append(history_message)
            while len(rollout_nodes) > 0 and rollout_iter_index < _args.max_rollout_iterations:
                infer_requests = [InferRequest([system_message, prompt_message, h, next_message]) for h in history_messages]
                # Because template will pop out last assistant message, so add an additional one.
                responses = self.infer_engine.infer(infer_requests, rollout_request_config, **self.infer_kwargs)
                rollout_iter_index += 1
                rollout_node_index = 0
                for index, response in enumerate(responses):
                    for key, value in self.usage_info.__dict__.items():
                        update_value = getattr(response.usage, key, None) + value
                        setattr(self.usage_info, key, update_value)
                    output = response.choices[0].message.content.rstrip(SEP_TOKEN + '\n')
                    output = output.split(SEP_TOKEN)[0]
                    output += SEP_TOKEN + '\n'
                    history_messages[rollout_node_index]["content"] += output
                    end_path = history_messages[rollout_node_index]["content"]
                    if check_terminate(end_path)[0]:
                        orm_infer_requests = [InferRequest([history_messages[rollout_node_index]])]
                        orm_score, _orm_mask = get_reward(
                            self.orm_model, orm_infer_requests, ground_truths=[ground_truth] * len(infer_requests),
                            threshold=0.0)
                        node.active_children[index].outcome_reward = orm_score[0]
                        if orm_score[0] == 1:
                            correct_answers.append(end_path)
                        else:
                            incorrect_answers.append(end_path)
                        rollout_nodes.pop(rollout_node_index)
                        history_messages.pop(rollout_node_index)
                        rollout_node_index -= 1
                    rollout_node_index += 1

        def _back_propagate(curr_node: LanguageNode):
            while curr_node:
                best_child_value = max([child.outcome_reward for child in curr_node.children])
                curr_node.init_and_update_value(best_child_value)
                curr_node.visit()
                curr_node = curr_node.parent

        def _collect(curr_node: LanguageNode):
            if curr_node.is_leaf():
                return []
            results = []
            for child in curr_node.children:
                results += _collect(child)
            curr_node.children = sorted(curr_node.children)
            if curr_node.children[-1].outcome_reward - curr_node.children[0].outcome_reward > 0.6:
                results.append(json.dumps({
                    "query": query,
                    "path": curr_node.path,
                    "good": curr_node.children[-1].path[-1],
                    "good_score": curr_node.children[-1].outcome_reward,
                    "bad": curr_node.children[0].path[-1],
                    "bad_score": curr_node.children[0].outcome_reward,
                }, ensure_ascii=False) + '\n')
            return results

        _args = self.args
        request_config = _args.get_request_config()
        request_config.stop = [SEP_TOKEN]
        request_config.seed = _args.seed
        _root = LanguageNode()

        correct_answers, incorrect_answers, prefer_pair = [], [], []
        terminate_correct, terminate_incorrect = [], []
        too_easy, too_hard = False, False
        iter_count = 0
        while (not too_easy and not too_hard
            and len(terminate_incorrect) + len(terminate_correct) < _args.num_return_sequences
            and iter_count < _args.max_iterations):
            logger.info(f"iter_count: {iter_count}" + "." * 10)
            logger.info("select" + "=" * 10)
            curr_node = _select(_root)
            logger.info("expand" + "=" * 10)
            _expand(curr_node)
            if curr_node.depth > 3:
                logger.info("rollout" + "=" * 10)
                _rollout(curr_node)
                logger.info("back propagate" + "=" * 10)
                _back_propagate(curr_node)
            if len(correct_answers) + len(incorrect_answers) >= _args.num_return_sequences:
                if 4 * len(incorrect_answers) < len(correct_answers):
                    logger.info("too easy" + "!" * 20)
                    logger.info(f"correct_answers: {correct_answers}")
                    logger.info(f"incorrect_answers: {incorrect_answers}")
                    too_easy = True
                elif 4 * len(correct_answers) < len(incorrect_answers):
                    logger.info("too hard" + "!" * 20)
                    logger.info(f"correct_answers: {correct_answers}")
                    logger.info(f"incorrect_answers: {incorrect_answers}")
                    too_hard = True
            iter_count += 1
        if iter_count == _args.max_iterations:
            logger.info("too hard" + "!" * 20)
            logger.info(f"correct_answers: {correct_answers}")
            logger.info(f"incorrect_answers: {incorrect_answers}")
            too_hard = True
        if not too_easy and not too_hard:
            prefer_pair = _collect(_root)
            logger.info(f"prefer_pair: {prefer_pair}")
        return prefer_pair

    def do_sample(self, data):
        if not isinstance(data, list):
            data = [data]
        generated = []
        for item in data:
            messages = item['messages'][0]
            query = messages[0]['content']
            ground_truth = messages[1]['content']
            prefer_pairs = self.search_single(query, ground_truth)
            generated += prefer_pairs
        return generated