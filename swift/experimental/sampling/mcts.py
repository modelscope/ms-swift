from copy import deepcopy
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from swift.llm import InferRequest
from swift.llm.infer.protocol import UsageInfo
from swift.utils import get_logger

from .base import Sampler
from .utils import get_reward, perform_infer
from .sampling_args import SamplingArguments

from typing import Union, List


logger = get_logger()

NXT_PROMPT = """Continue.
"""

next_message = {
    "role": "user",
    "content": NXT_PROMPT,
}

def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
    if isinstance(answers, str):
        answers = [answers]
    results = []
    for answer in answers:
        results.append("\\boxed" in answer)
    return results

class LanguageNode:

    def __init__(self,
                 step: str = None,
                 sep_token: str = None,
                 parent: "LanguageNode" = None,):
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
            self.infer_engines = [InferClient(base_url=base_url, api_key=api_key) for _ in range(args.num_return_sequences)]
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
        for i in range(_args.num_return_sequences):
            expand_request_config = deepcopy(request_config)
            expand_request_config.n = 1
            expand_request_config.num_beams = expand_request_config.n
            expand_request_config.seed += i
            self.expand_request_configs.append(expand_request_config)
        self.rollout_request_config = deepcopy(request_config)
        self.rollout_request_config.max_tokens = 500
        self.rollout_request_config.temperature = 0.0
        self.rollout_request_config.n = 1

    def update_usage_info(self, response):
        for key, value in self.usage_info.__dict__.items():
            update_value = getattr(response.usage, key, None) + value
            setattr(self.usage_info, key, update_value)

    def search_single(self, query, ground_truth):
        def _UCT(node: LanguageNode):
            alpha = _args.process_reward_rate
            value = alpha * node.process_reward + (1 - alpha) * node.outcome_reward
            if node.is_root():
                return value

            exploitation_score = value
            exploration_score = (_args.exploration_rate
                                 * np.sqrt(np.log(node.parent.visit_count + 1) / (node.visit_count + 1)))

            return exploration_score + exploitation_score

        def _select(node: LanguageNode):
            while not node.is_leaf():
                node = max(node.active_children, key=lambda x: _UCT(x))
            return node

        def _expand(node: LanguageNode):
            if node.is_root():
                infer_request = InferRequest([system_message, prompt_message])
            else:
                history_message = {
                    "role": "assistant",
                    "content": node.answer,
                }
                infer_request = InferRequest([system_message, prompt_message, history_message, next_message])

            # e_time = time.time()
            # 为了并行进行 Expand 操作，这里暂时不需要考虑顺序，因为 Prompt 是一样的
            n = _args.num_return_sequences - len(node.children)
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = {executor.submit(perform_infer,
                                           self.infer_engines[i],
                                           infer_request,
                                           self.expand_request_configs[i],
                                           **self.infer_kwargs): i for i in range(n)}
                responses = []
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        responses += future.result()
                    except Exception as e:
                        print(f"任务 {task_id} 执行请求时发生错误: {e}")
            # logger.info(f"expand.expand time: {time.time() - e_time}")

            # 为了并行获取 Outcome Reward，这里获得的 OR 是顺序返回的，所以可以直接对应
            orm_infer_requests = []
            unique_output = set() # 用于去重
            all_child_terminated = True
            for response in responses:
                self.update_usage_info(response)
                output = response.choices[0].message.content.rstrip(sep_token + '\n').split(sep_token)[0]
                if output in unique_output:
                    continue
                unique_output.add(output)
                orm_infer_requests.append(InferRequest([{"role": "assistant", "content": output}]))
                child = LanguageNode(step=output, parent=node)
                if check_terminate(child.answer)[0]:
                    child.terminated = True
                else:
                    all_child_terminated = False
                node.add_child(child)
            if all_child_terminated:
                node.terminated = True
                if not node.is_root():
                    node.parent.active_children.remove(node)

            # e_time = time.time()
            orm_score, _orm_mask = get_reward(
                self.orm_model, orm_infer_requests, ground_truths=[ground_truth] * len(orm_infer_requests),
                threshold=0.0)
            # logger.info(f"expand.orm time: {time.time() - e_time}")
            for child, score in zip(node.children, orm_score):
                if child.terminated:
                    child.init_and_update_value(score)
                    if child.outcome_reward == 1:
                        terminate_correct.append(child.answer)
                    else:
                        terminate_incorrect.append(child.answer)

            # e_time = time.time()
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
            # logger.info(f"expand.prm time: {time.time() - e_time}")

        def _rollout(node: LanguageNode):
            rollout_iter_index = 0
            rollout_nodes = {}
            for i in range(len(node.active_children)):
                rollout_nodes[i] = {
                    "node": node.active_children[i],
                    "history_messages": {
                        "role": "assistant",
                        "content": node.active_children[i].answer,
                    },
                }
            active_rollout_nodes = list(rollout_nodes.keys())
            while len(active_rollout_nodes) > 0 and rollout_iter_index < _args.max_rollout_iterations:
                # r_time = time.time()
                infer_requests = [InferRequest([system_message,
                                                prompt_message,
                                                rollout_nodes[index]['history_messages'],
                                                next_message])
                                  for index in active_rollout_nodes]
                # logger.info(f"rollout.prepare time: {time.time() - r_time}")
                # r_time = time.time()
                n = len(infer_requests)
                with ThreadPoolExecutor(max_workers=n) as executor:
                    futures = {executor.submit(perform_infer,
                                               self.infer_engines[i],
                                               infer_requests[i],
                                               self.rollout_request_config,
                                               **self.infer_kwargs): i for i in range(n)}
                    responses = []
                    for future in as_completed(futures):
                        task_id = futures[future]
                        try:
                            responses += future.result()
                        except Exception as e:
                            print(f"任务 {task_id} 执行请求时发生错误: {e}")
                # logger.info(f"rollout.infer time: {time.time() - r_time}")

                # r_time = time.time()
                orm_infer_requests = []
                end_paths = []
                for index, response in zip(active_rollout_nodes, responses):
                    self.update_usage_info(response)
                    output = response.choices[0].message.content.rstrip(sep_token + '\n').split(sep_token)[
                                 0] + sep_token + '\n'
                    rollout_nodes[index]['history_messages']["content"] += output
                    end_paths.append(rollout_nodes[index]['history_messages']["content"])
                    orm_infer_requests.append(InferRequest([rollout_nodes[index]['history_messages']]))
                # logger.info(f"rollout.orm_prepare time: {time.time() - r_time}")

                # r_time = time.time()
                orm_score, _orm_mask = get_reward(
                    self.orm_model, orm_infer_requests, ground_truths=[ground_truth] * len(infer_requests),
                    threshold=0.0)
                # logger.info(f"rollout.get_orm tiem: {time.time() - r_time}")
                terminated_state = check_terminate(end_paths)
                for index, score, terminated in zip(active_rollout_nodes, orm_score, terminated_state):
                    if terminated:
                        node.active_children[index].outcome_reward = score
                        if score == 1:
                            correct_answers.append(rollout_nodes[index]['history_messages']["content"])
                        else:
                            incorrect_answers.append(rollout_nodes[index]['history_messages']["content"])
                        rollout_nodes.pop(index)
                active_rollout_nodes = list(rollout_nodes.keys())
                rollout_iter_index += 1

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
                    "ground_truth": ground_truth,
                    "path": curr_node.path,
                    "good": curr_node.children[-1].path[-1],
                    "good_score": curr_node.children[-1].outcome_reward,
                    "bad": curr_node.children[0].path[-1],
                    "bad_score": curr_node.children[0].outcome_reward,
                }, ensure_ascii=False) + '\n')
            return results

        _args = self.args
        system_message = _args.system_message
        sep_token = _args.stop_words[0]
        _root = LanguageNode(sep_token=sep_token)
        prompt_message = {
            "role": "user",
            "content": query,
        }

        correct_answers, incorrect_answers, prefer_pair = [], [], []
        terminate_correct, terminate_incorrect = [], []
        too_easy, too_hard = False, False
        iter_count = 0
        while (not too_easy and not too_hard
            and len(terminate_incorrect) + len(terminate_correct) < _args.num_return_sequences
            and iter_count < _args.max_iterations):
            logger.info(f"iter_count: {iter_count}" + "." * 10)
            s_time = time.time()
            curr_node = _select(_root)
            logger.info("select" + "=" * 10+ f"time: {time.time() - s_time}")
            s_time = time.time()
            _expand(curr_node)
            logger.info("expand" + "=" * 10 + f"time: {time.time() - s_time}")
            if curr_node.depth > 3:
                s_time = time.time()
                _rollout(curr_node)
                logger.info("rollout" + "=" * 10 + f"time: {time.time() - s_time}")
                s_time = time.time()
                _back_propagate(curr_node)
                logger.info("back propagate" + "=" * 10 + f"time: {time.time() - s_time}")
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
            logger.info(f"time: {time.ctime(time.time())}")
            try:
                messages = item['messages'][0]
                query = messages[0]['content']
                ground_truth = messages[1]['content']
                prefer_pairs = self.search_single(query, ground_truth)
                generated += prefer_pairs
            except Exception as e:
                logger.error(f"Error: {e}")
        return generated