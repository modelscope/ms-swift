import json


def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
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
            rougel = evaluate_rougel([ref_input_json], [cand_input_json])
            if rougel is None or rougel < 10:
                f1.append(0)
            elif 10 <= rougel < 20:
                f1.append(0.1)
            else:
                f1.append(1)
        else:
            if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                # This cannot be happen, but:
                # line 62, in evaluate_action_reward
                # for k, v in ref_input_json.items():
                # AttributeError: 'str' object has no attribute 'items'
                # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
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
        return True
    else:
        return False


def parse_action(text):
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


def parse_output(text):
    action, action_input = parse_action(text)
    return action, action_input


def get_reward_toolbench(responses, ground_truths):
    rewards = []
    for ground_truth, response in zip(ground_truths, responses):
        action_ref = []
        action_input_ref = []
        action_pred = []
        action_input_pred = []
        reference = ground_truth
        prediction = response
        # prediction = self.tokenizer.decode(prediction)
        prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
        ref_action, ref_input = parse_output(reference)
        pred_action, pred_input = parse_output(prediction)
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

        reward = evaluate_action_reward(action_pred,
                                             action_ref,
                                             action_input_pred,
                                             action_input_ref
                                             )
        rewards.append(reward)
    return rewards


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return None
    try:
        from rouge import Rouge
        rouge = Rouge()
        rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
        rougel = rouge_score["rouge-l"]["f"]
        return rougel
    except:
        return None


orms = {
    'toolbench': get_reward_toolbench,
}