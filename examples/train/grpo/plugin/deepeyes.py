# some code borrowed from https://github.com/Visual-Agent/DeepEyes/blob/main

import base64
import io
import os
import re
from math import ceil, floor
from typing import Any, Dict, List
import random

import json
from openai import OpenAI
from PIL import Image

from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.orm import ORM

try:
    from math_verify import parse, verify
except ImportError as e:
    raise ImportError("please install math_verify by `pip install math_verify`") from e
"""
3 dataset file

    1. data_thinklite_reasoning_acc:  data_source == 'thinklite_eureka'
    2. data_0.1.2_visual_toolbox_v2.parquet : data_source == 'thinklite_eureka'

plugin:
    1. Tool to resize and rotate
    2. construct interleaved image-conv
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
{pred_ans}"""

# Whether to use the model for verification
# set the url to call the model in DeepEyesReward.__init__
USE_GEN_VERIFY = True


def load_pil_image(img):
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, str):
        if os.path.exists(img):
            return Image.open(img)

        try:
            if ',' in img:
                img_data = img.split(',')[1]
            else:
                img_data = img
            img_bytes = base64.b64decode(img_data)
            return Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise e

    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img))

    if hasattr(img, 'read'):
        return Image.open(img)


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


class DeepEyesReward(ORM):

    def __init__(self):
        super().__init__()
        if USE_GEN_VERIFY:
            try:
                self.client = OpenAI(
                    api_key='EMPTY',
                    base_url=f'http://127.0.0.1:8000/v1',
                )
                self.verify_model_name = self.client.models.list().data[0].id

    def __call__(self, completions, solution, extra_info, **kwargs) -> List[float]:
        # reference: https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py
        questions = extra_info['question']
        rewards = []
        for completion, sol, question in zip(completions, solution, questions):
            is_format_error = False
            # completion = "<think>" + completion
            count_think_1 = completion.count("<think>")
            count_think_2 = completion.count("</think>")
            if count_think_1 != count_think_2:
                is_format_error = True

            model_answer = ""
            predict_no_think = completion.split('</think>')[-1].strip()
            answer_pattern = r'\\boxed{([^}]+)}'
            answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
            if len(answer_list) == 0:
                acc_reward = 0.0
                is_format_error = True
            else:
                if len(answer_list) > 1:
                    is_format_error = True

                model_answer = answer_list[-1]
                if rule_math_verify(sol, model_answer):
                    acc_reward = 1.0
                else:
                    model_verify_reward = 0
                    if USE_GEN_VERIFY:
                        model_verify_reward = float(self.generative_verify(question, sol, model_answer))
                    
                    acc_reward = max(acc_reward, model_verify_reward)

            format_reward = -1.0 if is_format_error else 0.0
            reward = 1.2 * acc_reward + 0.4 * format_reward
            rewards.append(reward)

        return rewards

    def generative_verify(self, query, ground_truth, model_answer):
        full_prompt = MATH_VERIFY_PROMPT.format(
            query=query,
            gold_ans=ground_truth,
            pred_ans=model_answer,
        )

        chat_messages = [{"role": "user", "content": full_prompt}]
        response = ""
        for _ in range(8):
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.verify_model_name,
                    messages=chat_messages,
                    seed = random.randint(0, 1000000),
                    temperature=0.0,
                )
                response = chat_response.choices[0].message.content.strip()
                break
            except Exception as e:
                continue

        judgement = response.split('## Equivalence Judgement')[-1].lower()
        return 'true' in judgement and 'false' not in judgement


class VisualToolBoxScheduler(MultiTurnScheduler):
    name = 'visual_toolbox_v2'
    user_prompt = '\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> '

    def __init__(self, max_turns=None, *args, **kwargs):
        super().__init__(max_turns, *args, **kwargs)

    def check_finished(self, infer_request, result, current_turn):
        last_completion = infer_request.messages[-1]['content']
        answer = self.extract_answer(last_completion)
        if answer:
            return True
        action = self.extract_action(last_completion)
        if not action:
            return True

        return super().check_finished(infer_request, result, current_turn)

    def step(self, infer_request, result, current_turn):
        completion = result.message.content
        action = self.extract_action(completion)
        cropped_img = None
        try:
            tool_call = json.loads(action.strip())
            tool_name = tool_call['name']
            if tool_name != 'image_zoom_in_tool':
                raise ValueError(f'Unknown tool name: {tool_name}')
            args = tool_call['arguments']
            bbox = args['bbox_2d']
            img = infer_request.images[0]
            if not isinstance(img, Image.Image):
                img = load_pil_image(img)
                infer_request.images[0] = img
            origin_height = img.height
            origin_width = img.width
            bbox = self.maybe_resize_bbox(*bbox, origin_width, origin_height)
            cropped_img = img.crop(bbox)
            query = '<tool_response>' + '<image>' + self.user_prompt + '</tool_response>' + '<|im_end|>\n<|im_start|>assistant\n'
        except Exception as e:
            error_msg = f'Invalid tool call format: {action.strip()}. Error: {e}'
            query = f'Error: {str(error_msg)}'

        infer_request.messages.append({'role': 'user', 'content': query})
        if cropped_img:
            infer_request.images.append(cropped_img)

        return infer_request

    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height,
                                            width) <= 100, f'aspect ratio error: {left=}, {top=}, {right=}, {bottom=}'
            assert min(height, width) > 30, f'{height=}, {width=} is too small'
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False

    def maybe_resize_bbox(self, left, top, right, bottom, origin_width, origin_height):
        left = max(0, left)
        top = max(0, top)
        right = min(origin_width, right)
        bottom = min(origin_height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

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
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


multi_turns['deepeyes_scheduler'] = VisualToolBoxScheduler
