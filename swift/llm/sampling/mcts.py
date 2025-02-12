import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import json
import numpy as np

from swift.llm import InferRequest, SamplingArguments
from swift.llm.infer.protocol import UsageInfo
from swift.utils import get_logger
from .base import Sampler
from .utils import get_reward, perform_infer

logger = get_logger()

NXT_PROMPT = """Continue.
"""

next_message = {
    'role': 'user',
    'content': NXT_PROMPT,
}


class LanguageNode:

    def __init__(
        self,
        step: str = None,
        sep_token: str = None,
        parent: 'LanguageNode' = None,
    ):
        self.parent = parent

        if sep_token:
            self.sep_token = sep_token
        else:
            self.sep_token = parent.sep_token

        if parent:
            self.path = parent.path[:] + [step]
            self.answer = parent.answer + step + self.sep_token
            self.depth = parent.depth + 1
        else:
            self.path = []
            self.answer = ''
            self.depth = 0

        self.active_children = []
        self.children = []
        self.visit_count = 0
        self.process_reward = 0.0
        self.outcome_reward = 0.0
        self.terminated = False
        self.correct = False

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def visit(self):
        self.visit_count += 1

    def init_and_update_value(self, value):
        self.outcome_reward = (self.outcome_reward * self.visit_count + value) / (self.visit_count + 1)

    def add_child(self, child: 'LanguageNode'):
        self.children.append(child)
        if not child.terminated:
            self.active_children.append(child)

    def collect(self):
        result = {
            'path': self.path,
            'depth': self.depth,
            'visit_count': self.visit_count,
            'process_reward': self.process_reward,
            'outcome_reward': self.outcome_reward,
            'terminated': str(self.terminated),
            'correct': str(self.correct),
            'children': [child.collect() for child in self.children],
        }
        return result

    def __lt__(self, other):
        return self.outcome_reward < other.outcome_reward


class MctsSampler(Sampler):

    def __init__(self, input_args: SamplingArguments):
        super().__init__(input_args)
        self.usage_info = UsageInfo(0, 0, 0)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.infer_kwargs = {}
        if args.sampler_engine == 'client':
            from swift.llm import InferClient
            api_key = args.api_key
            base_url = args.base_url
            self.infer_engine = [
                InferClient(base_url=base_url, api_key=api_key) for _ in range(args.num_return_sequences)
            ]
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
        # Hack from super()
        self._prepare_request_configs()

    def _prepare_request_configs(self):
        _args = self.args
        request_config = _args.get_request_config()
        request_config.stop = _args.stop_words
        request_config.seed = _args.seed
        self.expand_request_configs = []
        self.rollout_request_configs = []
        for i in range(_args.num_return_sequences):
            expand_request_config = deepcopy(request_config)
            expand_request_config.n = 1
            expand_request_config.num_beams = expand_request_config.n
            expand_request_config.seed += i
            self.expand_request_configs.append(expand_request_config)
            rollout_request_config = deepcopy(request_config)
            rollout_request_config.max_tokens = 500
            rollout_request_config.temperature = 0.0
            rollout_request_config.n = 1
            self.rollout_request_configs.append(rollout_request_config)

    def update_usage_info(self, response):
        for key, value in self.usage_info.__dict__.items():
            update_value = getattr(response.usage, key, None) + value
            setattr(self.usage_info, key, update_value)

    def search_single(self, query, ground_truth):

        def _uct(uct_curr_node: LanguageNode):
            alpha = _args.process_reward_rate
            value = alpha * uct_curr_node.process_reward + (1 - alpha) * uct_curr_node.outcome_reward
            if uct_curr_node.is_root():
                return value

            exploitation_score = value
            exploration_score = (
                _args.exploration_rate
                * np.sqrt(np.log(uct_curr_node.parent.visit_count + 1) / (uct_curr_node.visit_count + 1)))

            return exploration_score + exploitation_score

        def _select(select_curr_node: LanguageNode):
            while not select_curr_node.is_leaf():
                select_curr_node = max(select_curr_node.active_children, key=lambda x: _uct(x))
            return select_curr_node

        def _expand(expand_curr_node: LanguageNode):
            n = _args.num_return_sequences - len(expand_curr_node.children)
            if expand_curr_node.is_root():
                infer_requests = [InferRequest(system_message + [prompt_message]) for _ in range(n)]
            else:
                history_message = {
                    'role': 'assistant',
                    'content': expand_curr_node.answer,
                }
                infer_request = InferRequest(system_message + [prompt_message, history_message, next_message])
                infer_requests = [infer_request for _ in range(n)]

            # e_time = time.time()
            # To perform the Expand operation in parallel,
            # there's no need to consider the order for now, since the Prompt is the same.
            expand_iter_index = 0
            while True:
                responses = perform_infer(self.infer_engine, infer_requests, self.expand_request_configs,
                                          **self.infer_kwargs)
                if len(responses) > 0:
                    break
                if expand_iter_index == 5:
                    raise ValueError('Expand should not return any response')
                expand_iter_index += 1
            # logger.info(f"expand.expand time: {time.time() - e_time}")

            # To fetch Outcome Reward in parallel,
            # the Outcome-Reward obtained is returned in order, so they can be directly matched accordingly.
            orm_infer_requests = []
            unique_output = set()
            for response in responses:
                self.update_usage_info(response)
                output = response.choices[0].message.content.rstrip(sep_token + '\n').split(sep_token)[0]
                if output in unique_output:
                    continue
                unique_output.add(output)
                orm_infer_requests.append(InferRequest([{'role': 'assistant', 'content': output}]))
                child = LanguageNode(step=output, parent=expand_curr_node)
                if self.orm_model.check_terminate(child.answer)[0]:
                    child.terminated = True
                expand_curr_node.add_child(child)

            # e_time = time.time()
            orm_score, _orm_mask = get_reward(
                self.orm_model,
                orm_infer_requests,
                ground_truths=[ground_truth] * len(orm_infer_requests),
                threshold=0.0)
            # logger.info(f"expand.orm time: {time.time() - e_time}")
            for child, score in zip(expand_curr_node.children, orm_score):
                if child.terminated:
                    child.init_and_update_value(score)
                    child.correct = score > 0.9
                    terminated_nodes.append(child)

            # e_time = time.time()
            if self.prm_model:
                prm_infer_requests = []
                for child in expand_curr_node.children:
                    prm_message = {'role': 'assistant', 'content': child.answer}
                    prm_infer_requests.append(InferRequest([prompt_message, prm_message]))
                prm_score, _prm_mask = get_reward(
                    self.prm_model,
                    prm_infer_requests,
                    ground_truths=[ground_truth] * len(prm_infer_requests),
                    threshold=0.0)
                for child, score in zip(expand_curr_node.children, prm_score):
                    child.process_reward = score
            # logger.info(f"expand.prm time: {time.time() - e_time}")

        def _rollout(rollout_curr_node: LanguageNode):
            rollout_depth = 0
            rollout_nodes = {}
            for i in range(len(rollout_curr_node.active_children)):
                rollout_nodes[i] = {
                    'node': rollout_curr_node.active_children[i],
                    'history_messages': {
                        'role': 'assistant',
                        'content': rollout_curr_node.active_children[i].answer,
                    },
                }
            active_rollout_nodes = list(rollout_nodes.keys())
            while len(active_rollout_nodes) > 0 and rollout_depth < _args.rollout_depth:
                # r_time = time.time()
                infer_requests = [
                    InferRequest(system_message
                                 + [prompt_message, rollout_nodes[index]['history_messages'], next_message])
                    for index in active_rollout_nodes
                ]
                # logger.info(f"rollout.prepare time: {time.time() - r_time}")
                # r_time = time.time()
                rollout_iter_index = 0
                while True:
                    responses = perform_infer(self.infer_engine, infer_requests, self.rollout_request_configs,
                                              **self.infer_kwargs)
                    if len(responses) > 0:
                        break
                    if rollout_iter_index == 5:
                        raise ValueError('Rollout should not return any response')
                    rollout_iter_index += 1
                # logger.info(f"rollout.infer time: {time.time() - r_time}")

                # r_time = time.time()
                orm_infer_requests = []
                end_paths = []
                for index, response in zip(active_rollout_nodes, responses):
                    self.update_usage_info(response)
                    output = response.choices[0].message.content.rstrip(sep_token
                                                                        + '\n').split(sep_token)[0] + sep_token + '\n'
                    rollout_nodes[index]['history_messages']['content'] += output
                    end_paths.append(rollout_nodes[index]['history_messages']['content'])
                    orm_infer_requests.append(InferRequest([rollout_nodes[index]['history_messages']]))
                # logger.info(f"rollout.orm_prepare time: {time.time() - r_time}")

                # r_time = time.time()
                orm_score, _orm_mask = get_reward(
                    self.orm_model,
                    orm_infer_requests,
                    ground_truths=[ground_truth] * len(infer_requests),
                    threshold=0.0)
                # logger.info(f"rollout.get_orm time: {time.time() - r_time}")
                terminated_state = self.orm_model.check_terminate(end_paths)
                for index, score, terminated in zip(active_rollout_nodes, orm_score, terminated_state):
                    if terminated:
                        rollout_curr_node.active_children[index].init_and_update_value(score)
                        if score > 0.9:
                            rollout_correct_answers.append(rollout_nodes[index]['history_messages']['content'])
                        else:
                            rollout_incorrect_answers.append(rollout_nodes[index]['history_messages']['content'])
                        rollout_nodes.pop(index)
                active_rollout_nodes = list(rollout_nodes.keys())
                rollout_depth += 1

        def _back_propagate(back_curr_node: LanguageNode):
            while back_curr_node:
                if back_curr_node == curr_node:
                    best_child_value = max([child.outcome_reward for child in back_curr_node.children])
                    back_curr_node.init_and_update_value(best_child_value)
                    last_child_value = back_curr_node.outcome_reward
                else:
                    back_curr_node.init_and_update_value(last_child_value)
                    last_child_value = back_curr_node.outcome_reward
                back_curr_node.visit()
                if len(back_curr_node.active_children) == 0:
                    back_curr_node.terminated = True
                    if not back_curr_node.is_root():
                        back_curr_node.parent.active_children.remove(back_curr_node)
                back_curr_node = back_curr_node.parent

        _args = self.args
        system_message = [] + _args.system_message
        sep_token = _args.stop_words[0] + '\n'
        _root = LanguageNode(sep_token=sep_token)
        prompt_message = {
            'role': 'user',
            'content': query,
        }

        rollout_correct_answers, rollout_incorrect_answers, terminated_nodes = [], [], []
        iter_count = 0
        stop_reason = None
        while True:
            logger.info(f'iter_count: {iter_count}' + '.' * 10)
            s_time = time.time()
            curr_node = _select(_root)
            logger.debug('select' + '=' * 10 + f'time: {time.time() - s_time}')
            s_time = time.time()
            _expand(curr_node)
            logger.debug('expand' + '=' * 10 + f'time: {time.time() - s_time}')
            if curr_node.depth > _args.rollout_start_depth:
                s_time = time.time()
                _rollout(curr_node)
                logger.debug('rollout' + '=' * 10 + f'time: {time.time() - s_time}')
            s_time = time.time()
            _back_propagate(curr_node)
            logger.debug('back propagate' + '=' * 10 + f'time: {time.time() - s_time}')
            if len(rollout_correct_answers) + len(rollout_incorrect_answers) >= 2 * _args.num_return_sequences:
                if 4 * len(rollout_incorrect_answers) < len(rollout_correct_answers):
                    stop_reason = 'too easy'
                    break
                elif 4 * len(rollout_correct_answers) < len(rollout_incorrect_answers):
                    stop_reason = 'too hard'
                    break
            if _root.terminated:
                stop_reason = 'root terminated'
                break
            if len(terminated_nodes) >= _args.num_return_sequences:
                stop_reason = 'enough nodes'
                break
            if iter_count >= _args.max_iterations:
                stop_reason = 'max_iterations'
                break
            iter_count += 1
        logger.info(f'stop_reason: {stop_reason}')
        # logger.info(f"rollout_correct_answers: {rollout_correct_answers}")
        # logger.info(f"rollout_incorrect_answers: {rollout_incorrect_answers}")

        monte_carlo_tree = _root.collect()
        result = {
            'query': query,
            'ground_truth': ground_truth,
            'rollout_correct_answers': rollout_correct_answers,
            'rollout_incorrect_answers': rollout_incorrect_answers,
            'monte_carlo_tree': monte_carlo_tree,
        }
        result_json = json.dumps(result, ensure_ascii=False)
        logger.info(result_json)
        return result_json

    def do_sample(self, data):
        if not isinstance(data, list):
            data = [data]
        generated = []
        for item in data:
            logger.info(f'time: {time.ctime(time.time())}')
            try:
                messages = item['messages'][0]
                query = messages[0]['content']
                ground_truth = messages[1]['content']
                generated.append(self.search_single(query, ground_truth) + '\n')
            except Exception as e:
                logger.error(f'Error: {e}')
                logger.error(f'Traceback: {traceback.format_exc()}')
        return generated
