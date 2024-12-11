# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import List, Union

from .tuner import prepare_model
from ..argument import RLFTArguments
from .sft import SwiftSft
from swift.utils import get_logger, get_model_parameter_info
from ...trainers import TrainerFactory

logger = get_logger()


class SwiftRLFT(SwiftSft):
    args_class = RLFTArguments
    args: args_class

    def get_reward(self, model, batch, query_responses, pad_token_id, context_length):
        scores_list = []
        socres_range_list = []
        for actions, action_inputs, query_response in zip(batch['actions'], batch['action_inputs'], query_responses):
            assert len(actions) == len(action_inputs)
            decoded_query_responses = self.template.processor.decode(query_response)
            from swift.llm.template import split_str_parts_by
            agent_parts = split_str_parts_by(decoded_query_responses, ['Action:', 'Action Input:', self.template.suffix[0]])
            agent_parts = [part for part in agent_parts if part['key'] in ['Action:', 'Action Input:']]

            ground_truths = []
            for action, action_input in (actions, action_inputs):
                ground_truths.append([action, action_input])

            agent_parts_processed = []
            temp_action = []
            for i, part in enumerate(agent_parts):
                if i % 2 == 0:
                    if part['key'] == 'Action:':
                        temp_action.append(part['content'])
                    else:
                        temp_action = []
                if i % 2 != 0:
                    if part['key'] == 'Action Input:':
                        temp_action.append(part['content'])
                    else:
                        temp_action = []

                    if len(part) == 2:
                        agent_parts_processed.append(temp_action)
                        temp_action = []

            scores = []
            score_ranges = []
            start_idx = 0
            for ground_truth, predition in (ground_truths, agent_parts):
                if not scores or scores[-1] == 1.0:
                    function_ok = ground_truth[0] == predition[0]
                    args_ok = self.compare_args(ground_truth[1], predition[1])
                    if function_ok and args_ok:
                        scores.append(1.0)
                    elif not function_ok and not args_ok:
                        scores.append(0.0)
                    else:
                        scores.append(0.1)
                else:
                    scores.append(0.0)

                action_str = f'Action:{predition[0]}'
                action_input_str = f'Action Input:{predition[1]}'
                action_ids = self.template.processor.encode(action_str, add_special_tokens=False)
                action_input_ids = self.template.processor.encode(action_input_str, add_special_tokens=False)
                start, end = self.find_sublist(query_response, action_ids)
                if start == -1 or end == -1:
                    scores = [scores[-1]]
                    score_ranges = [(-1, -1)]
                    break

                start_idx += end
                new_start, new_end = self.find_sublist(query_response[start_idx:], action_input_ids)
                if new_start == -1 or new_end == -1:
                    scores = [scores[-1]]
                    score_ranges = [(-1, -1)]
                    break

                score_ranges.append((start+start_idx, new_end+start_idx))
                start_idx += new_end

            scores_list.append(scores)
            socres_range_list.append(score_ranges)
        return scores_list, socres_range_list

    def find_sublist(self, full_list, sub_list):
        len_full = len(full_list)
        len_sub = len(sub_list)

        if len_sub > len_full:
            return -1, -1

        # 遍历主列表，查找子列表
        for i in range(len_full - len_sub + 1):
            # 提取主列表中的子片段
            current_segment = full_list[i:i + len_sub]

            # 比较当前片段是否与子列表相同
            if current_segment == sub_list:
                # 返回起始索引和结束索引（包含）
                return i, i + len_sub - 1

        # 如果未找到子列表，返回 (-1, -1)
        return -1, -1

    def compare_args(self, args1, args2):
        try:
            args1 = json.loads(args1)
        except Exception:
            pass

        try:
            args2 = json.loads(args2)
        except Exception:
            pass

        return args1 == args2

    def run(self):
        args = self.args

        train_dataset, val_dataset = self._get_dataset()
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        # Some tuners require train_dataset for preparation: LoRA-GA
        self.model = prepare_model(self.args, self.model)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        data_collator = self._get_data_collator()
        optimizers = self._get_optimizers(train_dataset)

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        reward_func_kwargs = {}
        if args.reward_type == 'agent':
            reward_func_kwargs = {'reward_func': self.get_reward}
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            optimizers=optimizers,
            template=self.template,
            **reward_func_kwargs,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def rlhf_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
