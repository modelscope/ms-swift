# Copyright (c) Alibaba, Inc. and its affiliates.
import json
from typing import List, Union
import torch
from rouge import Rouge
from swift.llm.train.tuner import prepare_adapter

from ..argument import RLFTArguments
from .sft import SwiftSft
from swift.utils import get_logger, get_model_parameter_info
from ...trainers import TrainerFactory

logger = get_logger()


class SwiftRLFT(SwiftSft):
    args_class = RLFTArguments
    args: args_class

    def evaluate_action_reward(self, action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = self.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # File "swift/swift/llm/train/rlft.py", line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return 2.0
        elif f1[0] == 0.0:
            return -1.0
        elif 0.0 < f1[0] < 1.0:
            return 0.5
        else:
            raise

    def parse_action(self, text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    def parse_output(self, text):
        action, action_input = self.parse_action(text)
        return action, action_input

    def get_reward(self, model, batch, query_responses, pad_token_id, context_length):
        rewards = []
        for ground_truth, query_response in zip(batch['ground_truth'], query_responses):
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = query_response[context_length:]
            prediction = self.tokenizer.decode(prediction)
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = self.parse_output(reference)
            pred_action, pred_input = self.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = self.evaluate_action_reward(action_pred,
                                                  action_ref,
                                                  action_input_pred,
                                                  action_input_ref
                                                  )
            rewards.append(reward)
        return None, torch.tensor(rewards, dtype=torch.float32).to('cuda'), None

    def evaluate_rougel(self, cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score["rouge-l"]["f"]
            return rougel
        except:
            return None

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
        from copy import deepcopy
        # Some tuners require train_dataset for preparation: LoRA-GA
        ref_model = deepcopy(self.model)
        self.model = prepare_adapter(self.args, self.model)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        data_collator = self._get_data_collator()
        optimizers = self._get_optimizers(train_dataset)
        from transformers import AutoModelForSequenceClassification
        value_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, trust_remote_code=True, num_labels=1
        )
        value_model.model_meta = self.model.model_meta
        value_model.model_info = self.model.model_info
        value_model.config.pad_token_id = self.template.processor.eos_token_id
        value_model = prepare_adapter(self.args, value_model, task='SEQ_CLS')
        # value_model.to('cuda:1')
        # from swift.utils.torch_utils import freeze_parameters
        # freeze_parameters(value_model, 0.85, freeze_parameters=[])
        # value_model.score.requires_grad = True
        trainer_cls = TrainerFactory.get_trainer_cls(args)
        reward_func_kwargs = {}
        if args.reward_type == 'agent':
            reward_func_kwargs = {'reward_func': self.get_reward}
        else:
            # Test code
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                args.model, trust_remote_code=True, num_labels=1
            )
            reward_func_kwargs = {
                'reward_model': reward_model
            }
        from copy import deepcopy
        trainer = trainer_cls(
            model=self.model,
            ref_model=ref_model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            value_model=value_model,
            callbacks=self.callbacks,
            optimizers=optimizers,
            template=self.template,
            **reward_func_kwargs,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def rlft_main(args: Union[List[str], RLFTArguments, None] = None):
    return SwiftRLFT(args).main()
