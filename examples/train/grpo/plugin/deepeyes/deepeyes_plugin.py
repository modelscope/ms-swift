# some code borrowed from https://github.com/Visual-Agent/DeepEyes/blob/main

import base64
import io
import os
import random
import re
from math import ceil, floor
from typing import Any, Dict, List

import json
from openai import OpenAI
from PIL import Image

from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.orm import ORM, orms

try:
    from math_verify import parse, verify
except ImportError as e:
    raise ImportError('please install math_verify by `pip install math_verify==0.5.2`') from e
"""
3 dataset file
    1. data_v0.8_visual_toolbox_v2.parquet:  data_source == 'chart' (vl_agent.compute_score)
    2. data_0.1.2_visual_toolbox_v2.parquet : data_source == 'vstar' (vl_agent.compute_score)
    3. data_thinklite_reasoning_acc.parquet: data_source == 'thinklite_eureka' (vl_agent.compute_score_math)

tool:
    image_zoom_in_tool: zoom in the image, return a cropped image
"""

MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""# noqa


def extract_answer(action_string: str) -> Dict[str, any]:
    answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
    return answer[-1] if answer else None


def extract_action(action_string: str) -> Dict[str, Any]:
    tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
    return tool_call_match[-1] if tool_call_match else None


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""# noqa
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def load_pil_image(img):
    try:
        if isinstance(img, Image.Image):
            return img

        elif isinstance(img, Dict):
            return Image.open(io.BytesIO(img['bytes']))

        elif isinstance(img, str):
            if os.path.exists(img):
                return Image.open(img)

            if ',' in img:
                img_data = img.split(',')[1]
            else:
                img_data = img
            img_bytes = base64.b64decode(img_data)
            return Image.open(io.BytesIO(img_bytes))

        elif isinstance(img, bytes):
            return Image.open(io.BytesIO(img))

        elif hasattr(img, 'read'):
            return Image.open(img)
        else:
            return img

    except Exception:
        return img


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


class DeepEyesReward(ORM):

    def __init__(self):
        super().__init__()
        try:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1',
            )
            self.verify_model_name = self.client.models.list().data[0].id
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy' or 'vllm serve'.") from e

    def __call__(self, completions, reward_model, extra_info, data_source, **kwargs) -> List[float]:
        # reference: https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py
        # NOTE: reward_model is a column name from the dataset, which contains the ground truth answer
        rewards = []
        messages = kwargs.get('messages')
        for completion, solution, info, source, message in zip(completions, reward_model, extra_info, data_source,
                                                               messages):
            sol = solution['ground_truth']
            info['messages'] = message
            if source in ['vstar', 'chart']:
                rewards.append(self.compute_score(completion, sol, info))
            elif source in ['thinklite_eureka']:
                rewards.append(self.compute_score_math(completion, sol, info))
            else:
                raise NotImplementedError

        return rewards

    def compute_score(self, predict_str: str, ground_truth: str, extra_info) -> float:
        is_format_error = False
        # predict_str = "<think>" + predict_str
        count_think_1 = predict_str.count('<think>')
        count_think_2 = predict_str.count('</think>')
        if count_think_1 != count_think_2:
            is_format_error = True
        count_tool_1 = predict_str.count('<tool_call>')
        count_tool_2 = predict_str.count('</tool_call>')
        if count_tool_1 != count_tool_2:
            is_format_error = True

        predict_no_think = predict_str.split('</think>')[-1].strip()
        count_answer_1 = predict_no_think.count('<answer>')
        count_answer_2 = predict_no_think.count('</answer>')
        if count_answer_1 != count_answer_2:
            is_format_error = True

        answer_text = predict_str.split('<answer>')[-1].split('</answer>')[0].strip()

        question_text = extra_info['question']
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        chat_response = self.client.chat.completions.create(
            model=self.verify_model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': full_prompt
                },
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()
        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()
            if '1' in response:
                acc_reward = 1.0
            elif '0' in response:
                acc_reward = 0.0
            else:
                acc_reward = 0.0
        else:
            if response == '1':
                acc_reward = 1.0
            elif response == '0':
                acc_reward = 0.0
            else:
                acc_reward = 0.0

        # Penalize for model trying to predict longer answer to hack llm-as-judge
        if len(answer_text) >= 1000:
            acc_reward = 0.0
            is_format_error = True

        num_image = 0
        for message in extra_info['messages']:
            if message['role'] == 'user' and '<image>' in message['content']:
                num_image += 1
        # More than one image indicates a successful tool call.
        tool_reward = 1.0 if num_image > 1 and acc_reward > 0.5 else 0.0
        format_reward = -1.0 if is_format_error else 0.0

        return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward

    def compute_score_math(self, predict_str: str, ground_truth: str, extra_info=None) -> float:
        is_format_error = False
        # predict_str = "<think>" + predict_str
        count_think_1 = predict_str.count('<think>')
        count_think_2 = predict_str.count('</think>')
        if count_think_1 != count_think_2:
            is_format_error = True

        model_answer = ''
        predict_no_think = predict_str.split('</think>')[-1].strip()
        answer_pattern = r'\\boxed{([^}]+)}'
        answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
        if len(answer_list) == 0:
            acc_reward = 0.0
            is_format_error = True
        else:
            if len(answer_list) > 1:
                is_format_error = True

            model_answer = answer_list[-1]
            if rule_math_verify(ground_truth, model_answer):
                acc_reward = 1.0
            else:
                acc_reward = 0
                full_prompt = MATH_VERIFY_PROMPT.format(
                    query=extra_info['question'],
                    gold_ans=ground_truth,
                    pred_ans=model_answer,
                )
                response = ''
                for _ in range(8):
                    try:
                        chat_response = self.client.chat.completions.create(
                            model=self.verify_model_name,
                            messages=[
                                {
                                    'role': 'user',
                                    'content': full_prompt
                                },
                            ],
                            seed=random.randint(0, 1000000),
                            temperature=0.0,
                        )
                        response = chat_response.choices[0].message.content.strip()
                        break
                    except Exception:
                        continue
                judgement = response.split('## Equivalence Judgement')[-1].lower()
                if 'true' in judgement and 'false' not in judgement:
                    acc_reward = 1.0

        format_reward = -1.0 if is_format_error else 0.0
        return 1.2 * acc_reward + 0.4 * format_reward


orms['deepeyes_reward'] = DeepEyesReward


class VisualToolBoxScheduler(MultiTurnScheduler):
    user_prompt = ('\nThink first, call **image_zoom_in_tool** if needed, then answer. '
                   'Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)'
                   '  <answer>...</answer> ')

    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        super().__init__(infer_engine, max_turns, *args, **kwargs)

    def check_finished(self, infer_request, response_choice, current_turn):
        should_stop = super().check_finished(infer_request, response_choice, current_turn)
        if should_stop:
            return True

        last_completion = infer_request.messages[-1]['content']

        action = extract_action(last_completion)
        # if the last completion is a tool call, do not finished yet
        if action:
            return False

        return True

    def step(self, infer_request, response_choice, current_turn):
        from qwen_vl_utils import fetch_image
        completion = response_choice.message.content
        action = extract_action(completion)
        cropped_img = None
        extra_info = {}
        try:
            tool_call = json.loads(action.strip())
            tool_name = tool_call['name']
            if tool_name != 'image_zoom_in_tool':
                raise ValueError(f'Unknown tool name: {tool_name}')
            args = tool_call['arguments']
            bbox = args['bbox_2d']
            # NOTE: this function is only compatible with the QwenVL series models
            # If you use another MLLM, please adjust the fetch_image function accordingly
            # ensure the returned img is of type PIL.Image.Image and
            # has been processed to a maximum size of max_pixels
            img = fetch_image({'image': load_pil_image(infer_request.images[0])})

            origin_height = img.height
            origin_width = img.width
            bbox = self.maybe_resize_bbox(bbox=bbox, origin_width=origin_width, origin_height=origin_height)
            # for invalid bbox, the exception will be catched in except block
            cropped_img = img.crop(bbox)
            query = '<tool_response>' + '<image>' + self.user_prompt + '</tool_response>'
        except Exception as e:
            error_msg = f'Invalid tool call format: {action.strip()}. Error: {e}'
            query = f'Error: {str(error_msg)}'

        infer_request.messages.append({'role': 'user', 'content': query})
        if cropped_img:
            infer_request.images.append(cropped_img)
        # override the images
        extra_info['images'] = infer_request.images

        # Return dictionary format according to new MultiTurnScheduler interface
        return {'infer_request': infer_request, 'rollout_infos': extra_info}

    def validate_bbox(self, left, top, right, bottom):
        assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
        height = bottom - top
        width = right - left
        assert max(height, width) / min(height,
                                        width) <= 100, f'aspect ratio error: {left=}, {top=}, {right=}, {bottom=}'
        assert min(height, width) > 30, f'{height=}, {width=} is too small'
        return True

    def maybe_resize_bbox(self, bbox, origin_width, origin_height):
        left, top, right, bottom = bbox

        left = max(0, left)
        top = max(0, top)
        right = min(origin_width, right)
        bottom = min(origin_height, bottom)
        self.validate_bbox(left, top, right, bottom)

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            self.validate_bbox(new_left, new_top, new_right, new_bottom)
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


multi_turns['deepeyes_scheduler'] = VisualToolBoxScheduler
