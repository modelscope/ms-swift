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
    """
    DeepEyes 风格的奖励计算器：
    - 面向多模态指令跟随/工具使用场景，基于 LLM-as-a-judge 与规则校验生成奖励；
    - 非数学题（vstar/chart）通过 `compute_score` 调用通用判别模型核验答案一致性；
    - 数学题（thinklite_eureka）通过 `compute_score_math` 优先使用 `math_verify`，
      失败时回退到判别模型进一步核验等价性；
    - 同时对输出格式（<think>/<tool_call>/<answer> 等标签）与工具使用行为进行惩奖。
    """

    def __init__(self):
        """
        初始化判别用的 OpenAI 兼容客户端：
        - 连接本地/远端的推理服务（默认 http://127.0.0.1:8000/v1）；
        - 读取可用模型列表，并保存第一个模型的 id 作为判别模型名；
        - 若连接失败，抛出运行时异常，提示部署方式。
        """
        # 初始化父类（ORM 基类）
        super().__init__()
        try:
            # 构造 OpenAI 兼容客户端，使用占位 api_key 与本地服务地址
            self.client = OpenAI(
                api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1',
            )
            # 读取服务端模型列表，取第一个模型 id 作为判别模型名
            self.verify_model_name = self.client.models.list().data[0].id
        except Exception as e:
            # 无法连接判别模型服务，给出明确的部署指引
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy' or 'vllm serve'.") from e

    def __call__(self, completions, reward_model, extra_info, data_source, **kwargs) -> List[float]:
        """
        计算一批样本的奖励分数。

        参数：
        - completions: 模型生成的答案列表；
        - reward_model: 数据集中“标准答案”列（元素字典，含 'ground_truth'）；
        - extra_info: 附加信息列表（元素通常带 question/messages 等）；
        - data_source: 数据来源标识列表（决定使用何种评分路径）；
        - kwargs: 其他可选参数（此处使用 messages）。

        返回：
        - List[float]：与输入样本一一对应的奖励分数。
        """
        # 参考实现链接；注意 reward_model 为数据列名（含标准答案）
        # reference: https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py
        # NOTE: reward_model is a column name from the dataset, which contains the ground truth answer
        # 结果分数列表
        rewards = []
        # 透传上游 messages（每条样本一份对话上下文）
        messages = kwargs.get('messages')
        # 遍历批次样本，按 data_source 分路打分
        for completion, solution, info, source, message in zip(completions, reward_model, extra_info, data_source,
                                                               messages):
            # 取出标准答案文本
            sol = solution['ground_truth']
            # 将对应对话上下文合入 info，供下游使用
            info['messages'] = message
            # 非数学题使用通用打分；数学题使用数学打分
            if source in ['vstar', 'chart']:
                rewards.append(self.compute_score(completion, sol, info))
            elif source in ['thinklite_eureka']:
                rewards.append(self.compute_score_math(completion, sol, info))
            else:
                # 未支持的数据来源
                raise NotImplementedError

        # 返回按顺序对应的奖励分数
        return rewards

    def compute_score(self, predict_str: str, ground_truth: str, extra_info) -> float:
        """
        通用任务打分（非数学题）：
        - 基于格式一致性检查与判别模型（LLM-as-a-judge）核验答案是否与标准一致；
        - 对大量文本尝试“骗分”与格式错误进行惩罚；
        - 对调用视觉工具（产生多张 <image>）的行为给予额外奖励。
        """
        # 初始化格式错误标记
        is_format_error = False
        # 检查 <think> 标签是否成对出现
        # predict_str = "<think>" + predict_str
        count_think_1 = predict_str.count('<think>')
        count_think_2 = predict_str.count('</think>')
        if count_think_1 != count_think_2:
            is_format_error = True
        # 检查 <tool_call> 标签是否成对出现
        count_tool_1 = predict_str.count('<tool_call>')
        count_tool_2 = predict_str.count('</tool_call>')
        if count_tool_1 != count_tool_2:
            is_format_error = True

        # 在去除思考片段后，检查 <answer> 标签是否成对出现
        predict_no_think = predict_str.split('</think>')[-1].strip()
        count_answer_1 = predict_no_think.count('<answer>')
        count_answer_2 = predict_no_think.count('</answer>')
        if count_answer_1 != count_answer_2:
            is_format_error = True

        # 提取答案主体文本，供判别模型打分
        answer_text = predict_str.split('<answer>')[-1].split('</answer>')[0].strip()

        # 组装判别提示词：包含问题、标准答案与模型答案
        question_text = extra_info['question']
        full_prompt = get_prompt(answer_text, ground_truth, question_text)

        # 调用判别模型进行一致性判断
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
        # 解析判别模型输出，统一为 1/0 的判定
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

        # 过长答案可能是在“绕过判别”的尝试，强制记为格式错误与 0 分
        # Penalize for model trying to predict longer answer to hack llm-as-judge
        if len(answer_text) >= 1000:
            acc_reward = 0.0
            is_format_error = True

        # 统计对话中出现的 <image> 次数；多于一张意味着工具调用成功
        num_image = 0
        for message in extra_info['messages']:
            if message['role'] == 'user' and '<image>' in message['content']:
                num_image += 1
        # 工具奖励：需工具调用成功且基础一致性得分较高（>0.5）
        # More than one image indicates a successful tool call.
        tool_reward = 1.0 if num_image > 1 and acc_reward > 0.5 else 0.0
        # 格式奖励：格式错误则给负分
        format_reward = -1.0 if is_format_error else 0.0

        # 融合三路分数：一致性为主、格式惩罚、小幅工具奖励
        return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward

    def compute_score_math(self, predict_str: str, ground_truth: str, extra_info=None) -> float:
        """
        数学任务打分：
        - 从答案中提取 LaTeX 形式的 \boxed{...} 作为最终数值/表达式；
        - 优先使用 `math_verify` 做语义等价验证，不等价时回退 LLM 判别以容忍表达差异；
        - 对格式错误（<think> 不匹配、无/多于一个 boxed）进行惩罚。
        """
        # 初始化格式错误标记
        is_format_error = False
        # 检查 <think> 标签是否成对出现
        # predict_str = "<think>" + predict_str
        count_think_1 = predict_str.count('<think>')
        count_think_2 = predict_str.count('</think>')
        if count_think_1 != count_think_2:
            is_format_error = True

        # 解析输出中的 boxed 结果，若不存在或不唯一则视为格式问题
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

            # 取最后一个 boxed 作为最终答案
            model_answer = answer_list[-1]
            # 首选严格的数学等价验证
            if rule_math_verify(ground_truth, model_answer):
                acc_reward = 1.0
            else:
                # 不等价则回退到 LLM 判别（容忍表达差异）
                acc_reward = 0
                full_prompt = MATH_VERIFY_PROMPT.format(
                    query=extra_info['question'],
                    gold_ans=ground_truth,
                    pred_ans=model_answer,
                )
                response = ''
                # 多次重试以提升鲁棒性
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
                # 解析判别输出中的等价结论
                judgement = response.split('## Equivalence Judgement')[-1].lower()
                if 'true' in judgement and 'false' not in judgement:
                    acc_reward = 1.0

        # 组合最终分数：数学准确性权重更高；若格式错误则有负向奖励
        format_reward = -1.0 if is_format_error else 0.0
        return 1.2 * acc_reward + 0.4 * format_reward


orms['deepeyes_reward'] = DeepEyesReward


class VisualToolBoxScheduler(MultiTurnScheduler):
    user_prompt = ('\nThink first, call **image_zoom_in_tool** if needed, then answer. '
                   'Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)'
                   '  <answer>...</answer> ')

    def __init__(self, max_turns=None, *args, **kwargs):
        super().__init__(max_turns, *args, **kwargs)

    def check_finished(self, infer_request, result, current_turn):
        should_stop = super().check_finished(infer_request, result, current_turn)
        if should_stop:
            return True

        last_completion = infer_request.messages[-1]['content']

        action = extract_action(last_completion)
        # if the last completion is a tool call, do not finished yet
        if action:
            return False

        return True

    def step(self, infer_request, result, current_turn):
        """
        执行单步多轮工具链逻辑：
        - 从模型输出（result）中解析工具调用 `<tool_call>`；
        - 若调用的是 `image_zoom_in_tool`，则读取 `bbox_2d` 对原图进行裁剪；
        - 将裁剪出的图片追加到后续输入，并拼接工具响应提示，供下一轮对话继续推理；
        - 若解析或裁剪失败，返回错误信息以提示模型。

        参数：
        - infer_request: 包含当前消息与图像上下文的对象（含 images 与 messages）；
        - result: 本轮模型输出对象（含 message.content，即带 `<tool_call>` 的文本）；
        - current_turn: 当前轮次编号（此处不直接使用，但保留接口一致性）。

        返回：
        - (infer_request, extra_info)：更新后的请求（附加了工具响应和/或裁剪图）与额外信息。
        """
        # 延迟导入，仅在需要使用视觉相关工具时加载，避免无关场景的依赖开销
        from qwen_vl_utils import fetch_image
        # 读取本轮模型输出文本（可能包含 <tool_call>）
        completion = result.message.content
        # 提取最新 <tool_call> 内容（若无则返回 None）
        action = extract_action(completion)
        # 预置变量：裁剪图（无则保持 None），额外信息（用于后续评估或调试）
        cropped_img = None
        extra_info = {}
        try:
            # 解析 JSON 化的工具调用报文
            tool_call = json.loads(action.strip())
            # 工具名称检查，仅允许 image_zoom_in_tool
            tool_name = tool_call['name']
            if tool_name != 'image_zoom_in_tool':
                raise ValueError(f'Unknown tool name: {tool_name}')
            # 读取参数并解出二维 bbox
            args = tool_call['arguments']
            bbox = args['bbox_2d']
            # NOTE: 仅兼容 QwenVL 系列模型；
            # 若为其他 MLLM，请按需调整 fetch_image 的入参与返回图像类型（PIL.Image.Image）
            # 同时需确保图像已按 max_pixels 等策略处理
            img = fetch_image({'image': load_pil_image(infer_request.images[0])})

            # 记录原图尺寸，用于对 bbox 的边界裁剪与最小尺寸兜底
            origin_height = img.height
            origin_width = img.width
            # 将 bbox 裁剪到图像范围，并在必要时扩张（见 maybe_resize_bbox 内部注释）
            bbox = self.maybe_resize_bbox(bbox=bbox, origin_width=origin_width, origin_height=origin_height)
            # 若 bbox 非法，maybe_resize_bbox/validate_bbox 会抛出异常并被下方捕获
            cropped_img = img.crop(bbox)
            # 构造工具响应：返回新图像占位符，并附带用户提示，交给下一轮继续思考与作答
            query = '<tool_response>' + '<image>' + self.user_prompt + '</tool_response>'
        except Exception as e:
            # 若解析/校验/裁剪任一步骤失败，向模型返回错误信息，提醒其重新生成格式正确的调用
            error_msg = f'Invalid tool call format: {action.strip()}. Error: {e}'
            query = f'Error: {str(error_msg)}'

        # 将工具响应或错误提示，作为用户消息附加到消息序列中，触发下一轮模型回复
        infer_request.messages.append({'role': 'user', 'content': query})
        # 若裁剪图存在，则将其作为新的图像输入，参与下一轮推理
        if cropped_img:
            infer_request.images.append(cropped_img)
        # 把最新图像上下文挂到 extra_info，便于上层组件/评估器使用
        extra_info['images'] = infer_request.images
        # 返回更新后的请求对象与附加信息
        return infer_request, extra_info

    def validate_bbox(self, left, top, right, bottom):
        """
        校验二维边界框是否合法，若不合法则抛出 AssertionError。

        校验规则：
        - 形状有效：必须满足 left < right 且 bottom > top；
        - 纵横比限制：max(height, width) / min(height, width) ≤ 100，避免极端细长的裁剪框；
        - 尺寸下限：min(height, width) > 30 像素，避免过小区域影响可见性与稳定性。

        参数说明：
        - left: 左边界像素坐标；
        - top: 上边界像素坐标；
        - right: 右边界像素坐标；
        - bottom: 下边界像素坐标；

        返回：
        - True：所有断言通过时返回 True；否则抛出断言异常。

        备注：
        - 本方法由 `maybe_resize_bbox` 在裁剪前/后调用，用于确保传入的 bbox 可被 PIL.Image.crop 安全使用。
        """
        # 1) 断言边界顺序合法：左边应小于右边，顶部应小于底部
        assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
        # 2) 计算当前边界框的高与宽
        height = bottom - top
        width = right - left
        # 3) 控制极端宽高比，过滤异常的“超长条”形 bbox
        assert max(height, width) / min(height,
                                        width) <= 100, f'aspect ratio error: {left=}, {top=}, {right=}, {bottom=}'
        # 4) 限制最小边长度，避免裁剪区域过小
        assert min(height, width) > 30, f'{height=}, {width=} is too small'
        # 5) 若所有检查通过，返回 True
        return True

    def maybe_resize_bbox(self, bbox, origin_width, origin_height):
        """
        将传入的二维边界框裁剪到图像有效范围内，并在必要时按中心等比扩张，
        使最短边不少于 28 像素；在进入与（可能的）扩张后都会进行合法性校验。

        注意：当前 `validate_bbox` 要求最短边 > 30，故若外部阈值更严格，
        此处针对 <28 的兜底扩张分支通常不会触发；只有当 `validate_bbox` 的
        最小边阈值放宽（例如 ≤28）时，该分支才用于把过小 bbox 扩张到可用尺寸。

        参数：
        - bbox: [left, top, right, bottom] 边界框坐标（像素，可能为浮点数）；
        - origin_width: 原始图像宽；
        - origin_height: 原始图像高；

        返回：
        - [left, top, right, bottom]：裁剪且（可能）扩张后的 bbox；扩张分支使用
          floor/ceil 将边界对齐到整数像素。
        """
        # 解包输入的 bbox 坐标
        left, top, right, bottom = bbox

        # 将 bbox 裁剪到图像范围内，防止越界
        left = max(0, left)
        top = max(0, top)
        right = min(origin_width, right)
        bottom = min(origin_height, bottom)
        # 对裁剪后的 bbox 做一次合法性校验（形状、比例、最小尺寸）
        self.validate_bbox(left, top, right, bottom)

        # 计算当前 bbox 的高与宽
        height = bottom - top
        width = right - left
        # 若最短边小于 28，则按中心放大，保证最短边至少为 28
        if height < 28 or width < 28:
            # 计算 bbox 的中心点坐标
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            # 计算需要放大的比例，使 min(height, width) 放大到 28
            ratio = 28 / min(height, width)
            # 计算放大后的半高与半宽，使用 ceil 保证覆盖充足像素
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            # 根据中心点与半宽高计算新的边界，左右分别用 floor/ceil 对齐到整数像素
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            # 对扩张后的 bbox 再做一次合法性校验
            self.validate_bbox(new_left, new_top, new_right, new_bottom)
            # 返回扩张后的 bbox
            return [new_left, new_top, new_right, new_bottom]
        # 若无需扩张，返回裁剪后的原 bbox
        return [left, top, right, bottom]

multi_turns['deepeyes_scheduler'] = VisualToolBoxScheduler
