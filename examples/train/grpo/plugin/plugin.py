import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):
    """
    数学准确性奖励函数。

    使用 `math_verify` 与 `latex2sympy2_extended` 解析 LaTeX 表达式，对模型输出与标准答案进行符号级验证。
    若标准答案可被成功解析，则要求模型答案也需为合法 LaTeX 并与标准答案等价；
    若标准答案不可解析，则跳过该样本（返回 1.0，避免影响训练）。
    """

    def __init__(self):
        """初始化依赖检查，确保运行时具备 `math_verify` 包。"""
        # 动态检查第三方依赖是否安装
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        计算每条样本的奖励分数。

        参数:
            completions (List[str]): 模型输出的文本列表。
            solution (List[str]): 标准答案（LaTeX 形式）列表。
            **kwargs: 预留扩展参数，不使用。

        返回:
            List[float]: 与 `completions` 对齐的一维奖励分数列表，取值 {0.0, 1.0}。
        """
        # 引入所需解析与归一化配置
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []  # 累积每个样本的奖励
        for content, sol in zip(completions, solution):
            # 解析标准答案：仅当能成功解析时才进行严格验证
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # 要求模型答案提供为合法的 LaTeX（不允许畸形运算符）
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # 使用符号验证：等价返回 True -> 1.0，否则 0.0
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # 标准答案不可解析：该样本跳过，奖励置为 1.0（不惩罚模型）
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):
    """
    数学答案格式奖励函数。

    要求输出严格匹配 `<think>...</think>` 后紧跟 `<answer>...</answer>` 且无尾随内容。
    满足则返回 1.0，否则 0.0。
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查文本是否满足指定输出格式的奖励函数。

        要求格式：以 `<think>...</think>` 开头，紧随一个 `<answer>...</answer>`，且文本末尾无其它内容。

        参数:
            completions (List[str]): 模型输出文本列表。
            **kwargs: 预留扩展参数，不使用。

        返回:
            List[float]: 满足格式返回 1.0，否则 0.0。
        """
        # 正则: 
        # ^ 开头；<think>...</think> 任意多行；可有空白；<answer>...</answer>；(?![\s\S]) 断言后续为空（精确到结尾）
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        # 为每个输出执行多行/点任意匹配
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        # 命中返回 1.0，否则 0.0
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):
    """
    倒计时算式格式与正确性奖励函数。

    要求模型在 `<answer>...</answer>` 中给出等式；
    - 等式左边使用提供的所有数字且各用一次；
    - 等式仅包含数字、+ - * / 括号与空白；
    - 等式求值与给定目标 `gt` 在 1e-5 范围内相等。
    满足则奖励 1.0，否则 0.0。
    """

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        基于格式与数值正确性的倒计时题奖励。

        参数:
            completions (List[str]): 模型生成的文本列表。
            target (List[str]): 期望的数值结果列表。
            nums (List[str]): 可用数字集合（每条样本）。

        返回:
            List[float]: 每条样本的奖励分数（0.0 或 1.0）。
        """
        rewards = []  # 累积每个样本的奖励
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # 1) 提取 <answer> 标签内容
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # 2) 拿到等式主体，并去除等号右侧
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # 3) 提取使用到的数字
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # 4) 检查所有数字是否与提供的集合完全一致（数量一致、各一次）
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # 5) 只允许合法字符与算符
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # 6) 受限环境下求值（注意：eval 有风险，这里依赖上面的正则限制）
                result = eval(equation, {"__builti'ns__": None}, {})
                # 7) 校验结果与目标数值
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # 任意异常（解析/计算失败）均记 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):
    """
    多模态答案准确性奖励函数。

    优先使用符号验证（`math_verify`）判断等价；若失败，则降级为基于 `<answer>` 标签的字符串匹配。
    任一方法判定等价则记 1.0，否则 0.0。
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        校验答案正确性的奖励函数。
        参数:
            completions (List[str]): 模型输出文本。
            solution (List[str]): 标准答案文本。

        返回:
            List[float]: 逐样本奖励分数（0.0 或 1.0）。
        """
        rewards = []  # 累积每个样本奖励
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                # 忽略解析/验证异常，进入下一种比对方式
                pass

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    # 二次比对失败保持 0.0
                    pass
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):
    """
    基于 E2B 沙箱的代码执行奖励函数。

    流程：
    1) 从模型输出中抽取指定语言的代码块；
    2) 组装包含若干测试用例的评估脚本；
    3) 使用 E2B Code Interpreter 在沙箱中并发运行，返回各样本通过率。
    依赖项：`e2b-code-interpreter`、`.env`（如需凭据）。
    """

    def __init__(self):
        """
        初始化 CodeReward 运行所需的依赖与环境。
        此构造函数用于在运行前检查 `e2b-code-interpreter` 是否可用，并加载 `.env` 环境变量。
        """
        # 延迟导入以避免在未使用该奖励函数时强依赖
        import importlib.util
        # 断言 e2b 解释器可用，否则给出安装指引
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        # 读取 .env 以加载可能需要的鉴权、网络配置等
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        """
        从 Markdown 代码块中提取指定语言的最后一段代码。

        Args:
            completion (str): 模型的完整输出文本，可能包含多个代码块。
            language (str): 目标代码语言（例如 'python', 'python3', 'javascript'）。

        Returns:
            str: 提取到的代码字符串；若未找到匹配的代码块则为空字符串。
        """
        # 根据语言名构建匹配 Markdown 代码块的正则，如 ```python\n...```
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        # 在输出文本中查找所有匹配的代码段
        matches = pattern.findall(completion)
        # 取最后一个匹配的代码块（通常模型会在答案末尾给出最终代码）
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        # 返回代码字符串（可能为空串）
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """
        在同步环境中执行异步代码评估，并返回奖励列表。

        Args:
            scripts (List[str]): 针对每个样本构造的可执行评估脚本（字符串）。
            languages (List[str]): 对应每个脚本的语言标识（供 E2B 选择解释器）。

        Returns:
            List[float]: 各样本的评估得分（通过率 0~1）。
        """
        # 新建一个事件循环
        loop = asyncio.new_event_loop()
        # 将新建循环设置为当前线程的事件循环
        asyncio.set_event_loop(loop)

        try:
            # 在事件循环中运行异步评估，并等待其完成
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            # 无论是否出错，都要确保事件循环被关闭
            loop.close()

        # 返回各样本得分列表
        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        """
        异步并发执行所有评估脚本。

        Args:
            scripts (List[str]): 针对每个样本构造的可执行评估脚本。
            languages (List[str]): 与脚本一一对应的语言名称。

        Returns:
            List[float]: 各样本评估产生的得分（0~1）。
        """
        # 延迟导入 E2B 的异步沙箱类
        from e2b_code_interpreter import AsyncSandbox

        # 手动创建沙箱（该版本暂不提供异步上下文管理器）
        try:
            # 创建沙箱实例，设置整体超时与单次请求超时
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            # 创建沙箱失败，记录告警并返回与输入等长的 0.0 列表
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # 为每个脚本构建一个运行任务，以便并发执行
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # 并发等待所有任务完成，收集它们的返回值
        results = await asyncio.gather(*tasks)
        # 将返回的可迭代结果转为列表（显式拷贝）
        rewards = list(results)

        # 所有任务完成后释放沙箱资源
        await sbx.kill()

        # 返回奖励列表
        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        """
        在 E2B 沙箱中执行单个评估脚本，并将输出解析为浮点得分。

        Args:
            sbx: 已创建的 E2B 异步沙箱实例。
            script (str): 需执行的完整评估脚本（会在沙箱中运行）。
            language (str): 代码语言（用于选择解释器）。

        Returns:
            float: 该脚本对应样本的得分；无法解析或执行失败时返回 0.0。
        """
        try:
            # 在沙箱中执行脚本，限制单次执行的超时时间
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            # 执行阶段出现异常（网络、语法、沙箱问题等），记告警并返回 0 分
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            # 将沙箱返回的文本输出尝试解析为浮点数（预期为 0~1 的通过率）
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        评估代码型任务的模型输出，返回各样本通过率作为奖励。

        约定 `kwargs` 中包含 `verification_info`：
        - language: 代码语言（如 'python3'）
        - test_cases: 用例数组，每个包含 {input, output}

        Args:
            completions (List[str]): 模型输出文本集合，每个可能包含 Markdown 代码块。
            **kwargs: 需包含键 `verification_info`（List[Dict]）。

        Returns:
            List[float]: 各样本在其测试用例上的通过率（0~1）。
        """
        # 评估脚本模板：在沙箱内运行，被格式化注入 code 与 test_cases
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        # 从 kwargs 读取验证信息（包含语言与测试用例）
        verification_info = kwargs['verification_info']
        # 为每个样本提取语言，便于选择执行环境
        languages = [info['language'] for info in verification_info]
        # 从模型输出中抽取与语言对应的代码片段
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        # 基于模板为每个样本拼装完整的可执行评估脚本
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            # 在同步上下文中调用异步执行，获得各样本得分
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            # 任一阶段出错时，以 0.0 作为降级得分，保证训练过程不中断
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        # 返回最终的奖励分数
        return rewards


class CodeFormat(ORM):
    """
    代码格式合规奖励函数。

    要求输出满足：
      - 存在 `<think>...</think>`；
      - `<answer>...</answer>` 中包含与 `verification_info.language` 对应的 Markdown 代码块。
    满足则记 1.0，否则 0.0。
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        """按语言检查 `<answer>` 中是否含符合 Markdown 语法的代码块。"""
        # 读取每条样本对应的语言与其它验证信息
        verification_info = kwargs['verification_info']
        # 预备结果列表
        rewards = []  # 累积每个样本奖励
        # 遍历模型输出与对应的验证信息
        for content, info in zip(completions, verification_info):
            # 构造基于语言的正则：要求 <think> 段、<answer> 段，且 <answer> 中含 ```{language} ... ``` 代码块并精确到结尾
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            # 执行多行/点任意匹配
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            # 命中则奖励 1.0，否则 0.0
            reward = 1.0 if match else 0.0
            # 记录该样本奖励
            rewards.append(reward)
        # 返回全部样本奖励
        return rewards


class CodeRewardByJudge0(ORM):
    """
    基于 Judge0 在线评测的代码执行奖励函数。

    通过调用 Judge0 REST API 运行代码并核对期望输出，计算每条样本在测试用例上的通过率。
    需要设置环境变量：`JUDGE0_ENDPOINT`（必需）、`JUDGE0_X_AUTH_TOKEN`（可选）。
    """
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        """
        初始化 Judge0 客户端配置与请求头。

        环境变量:
            - JUDGE0_ENDPOINT: Judge0 服务的基础 API 地址（必填）。
            - JUDGE0_X_AUTH_TOKEN: 可选鉴权 Token（若服务端开启鉴权）。

        Returns:
            None
        """
        # 从环境读取 Judge0 API 地址（必需项）
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        # 未配置时直接报错，避免运行时隐式失败
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        # 可选的鉴权 Token（如果部署端启用鉴权）
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        # 基础请求头采用 JSON
        self.headers = {'Content-Type': 'application/json'}
        # 如有 Token，将其加入自定义请求头
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        """
        从 Markdown 代码块中抽取指定语言的最后一段代码。

        Args:
            completion (str): 模型完整输出，可能包含多个代码块。
            language (str): 代码语言（如 'python3'、'javascript'）。

        Returns:
            str: 提取到的代码片段；若不存在匹配代码块则返回空字符串。
        """
        # 为目标语言构造匹配 Markdown 代码块的正则
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        # 查找所有匹配的代码块内容
        matches = pattern.findall(completion)
        # 取最后一个匹配（通常答案末尾给出最终代码）
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        # 返回抽取结果
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        """
        将语言名称映射为 Judge0 的 `language_id`。

        Args:
            language (str | None): 代码语言名称；为 None 时默认 Python。

        Returns:
            int: Judge0 的语言标识。
        """
        if language is None:
            return cls.PYTHON_ID  # 默认使用 Python3
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        """
        调用 Judge0 执行单条样本的全部测试用例并返回通过率。

        Args:
            code (str): 待评测的源代码。
            test_cases (List[Dict]): 测试用例，元素包含键 'input' 与 'output'。
            language_id (int): Judge0 所需的语言标识。

        Returns:
            float: 该样本在所有用例上的通过率（0~1）。
        """
        import aiohttp  # 异步 HTTP 客户端
        try:
            passed = 0  # 通过的用例数
            total = len(test_cases)  # 用例总数

            for case in test_cases:
                # 空代码片段直接跳过执行
                if code is not None and code != '':
                    # 为每个用例创建独立的 HTTP 会话
                    async with aiohttp.ClientSession() as session:
                        # 组装 Judge0 提交所需的负载
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        # 同步等待评测完成（wait=true），便于直接读取结果
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            # 解析响应 JSON
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            # 仅在状态为 Accepted 时计为通过
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            # 计算通过率（避免除零，total 来自 len(test_cases) 且应 ≥1）
            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            # 发生网络或服务端错误时，返回 0 分并记录警告
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        """
        在同步上下文中执行异步评估流程并返回结果。

        Returns:
            List[float]: 全部样本的通过率列表。
        """
        # 创建隔离的事件循环，避免污染外部循环状态
        loop = asyncio.new_event_loop()
        # 将新事件循环绑定到当前线程
        asyncio.set_event_loop(loop)
        try:
            # 在事件循环内运行异步任务并等待其完成
            rewards = loop.run_until_complete(self.run_async())
        finally:
            # 关闭事件循环，释放资源
            loop.close()
        # 返回评估结果
        return rewards

    async def run_async(self):
        """
        并发评估所有样本，聚合返回通过率列表。

        Returns:
            List[float]: 全部样本的通过率。
        """
        # 为每个样本构建一个异步评测任务
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        # 并发调度所有任务并等待完成
        results = await asyncio.gather(*tasks)
        # 将结果转为列表形式
        rewards = list(results)
        # 返回最终结果
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        入口方法：解析代码、异步评测并返回各样本通过率。

        Args:
            completions (List[str]): 模型输出文本，可能包含多段 Markdown 代码。
            **kwargs: 需要包含键 `verification_info`，提供语言与测试用例。

        Returns:
            List[float]: 每条样本在其测试用例上的通过率（0~1）。
        """
        # 读取验证信息（每条样本包含语言与测试用例）
        self.verification_info = kwargs['verification_info']

        # 收集样本语言，便于后续选择 Judge0 语言 ID
        languages = [info['language'] for info in self.verification_info]
        # 从模型输出中按语言提取代码片段
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            # 启动异步评测流程（在同步上下文中执行）
            rewards = self.run_async_from_sync()
        except Exception as e:
            # 任意异常时降级为 0.0，避免训练过程失败
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        # 返回评测得分
        return rewards


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):
    """
    工具使用格式奖励函数。

    功能概述：
    - 基于标注 `solution` 指示的目标输出形态，对模型 `completions` 的实际输出进行格式校验并打分；
      受支持的目标形态：
        1) 仅 `<response>`（且必须有 `<think>` 作为前缀段）
        2) 仅 `<tool_call>`（且必须有 `<think>` 作为前缀段）
        3) `<tool_call>` 后紧跟 `<response>`（均需有且仅有一对标签，且必须由 `<think>` 开头）
        4) 仅 `<think>`（无 `<response>` 与 `<tool_call>`）

    动态范围：
    - 基础上下界由 `self.format_max_possible` 与 `self.format_min_possible` 给出；
    - 若设置环境变量 `MAX1STEP30MAX3=1`：当 `global_step >= 30` 时将上下界缩小为原来的一半；否则保持原值；
      （注意：此处行为以代码实现为准，与注释“step < 30 时缩放”不同）
    - 若设置环境变量 `SCHEDULEREWARD=1`：按 `global_step` 做线性插值调整上下界：
        max_possible = 2 - (2 - max_possible) * global_step / 150
        min_possible = -2 + (2 + min_possible) * global_step / 150
      并将上界下界分别截断至不低于 1.0 与不高于 -1.0（即最终 max>=1.0, min<=-1.0）。

    返回：
    - 对齐 `completions` 的奖励分数列表，每个样本在未命中目标格式时取动态区间的下界；
      满足目标格式且标签次数严格为 1 时取动态区间的上界。
    """
    def __init__(self):
        """
        初始化格式奖励的默认上下界。

        属性：
        - format_max_possible (float): 格式奖励的最大可能值（默认 1.0）。
        - format_min_possible (float): 格式奖励的最小可能值（默认 0.0）。
        """
        # 设置格式奖励的上界默认值
        self.format_max_possible = 1.0
        # 设置格式奖励的下界默认值
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        根据目标格式（由标注 `solution` 决定）为每条样本打分。

        参数：
        - completions (List[str]): 模型输出列表（每个元素为完整文本，含 `<think>` 段）。
        - solution (List[str]): 标注列表，用以指示目标输出形态（是否包含 `<tool_call>` / `<response>`）。
        - kwargs: 需要包含键 `trainer_state`，其属性 `global_step` 用于动态缩放奖励区间。

        环境变量：
        - MAX1STEP30MAX3=1：若启用且 `global_step >= 30`，将上下界缩小为原来的一半；否则保持原值。
        - SCHEDULEREWARD=1：在训练过程中按 `global_step` 对上下界进行线性插值，并裁剪至 [min<=-1.0, max>=1.0]。

        返回：
        - List[float]: 与 `completions` 对齐的奖励分数。
        """
        # 从 kwargs 读取训练状态对象（由上游训练器注入）
        trainer_state = kwargs.get('trainer_state')
        # 读取当前全局训练步数，用于动态缩放奖励区间
        global_step = trainer_state.global_step
        # 初始化当前样本的可达奖励上界与下界（后续可能被动态调整）
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two-stage 设置：依据环境变量将训练过程分段并缩放奖励区间
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            # 当步数达到阈值（>=30）时，将上下界缩小为原来的一半，否则保持原值
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # 连续插值：在训练过程中按步数对奖励区间做线性插值，并对边界进行裁剪
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            # 线性插值到一个以 2 与 -2 为锚点的区间，然后做截断
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            # 上界不低于 1.0
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            # 下界不高于 -1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        # 预备结果容器：累积各样本奖励
        rewards = []
        # 为可读性，重命名为 responses（模型的生成结果）
        responses = completions

        for response, ans in zip(responses, solution):
            # 默认给到动态区间的下界（未命中目标格式）
            reward = min_possible_reward
            # 目标仅包含 <response> 段（且前缀必须有 <think>）
            if '<response>' in ans and '<tool_call>' not in ans:
                # 要求：<think>...</think> 紧随 <response>...</response> 且文本到结尾；标签各出现 1 次
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response, re.DOTALL) \
                        and response.count('<response>') == 1 \
                        and response.count('</response>') == 1:
                    reward = max_possible_reward
            # 目标仅包含 <tool_call> 段（且前缀必须有 <think>）
            elif '<response>' not in ans and '<tool_call>' in ans:
                # 要求：<think>...</think> 紧随 <tool_call>...</tool_call> 且文本到结尾；标签各出现 1 次
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response, re.DOTALL) \
                        and response.count('<tool_call>') == 1 \
                        and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            # 目标同时包含 <tool_call> 与 <response>，且顺序为先 tool_call 后 response
            elif '<response>' in ans and '<tool_call>' in ans:
                # 要求：<think>...</think> 后先 <tool_call>...</tool_call> 再 <response>...</response> 且文本到结尾
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL)
                        and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1
                        and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            # 其余情况：仅允许 `<think>` 段
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            # 记录该样本的奖励
            rewards.append(reward)

        # 返回所有样本的奖励列表
        return rewards


class ToolUseLengthReward(ORM):
    """
    工具使用思考长度奖励函数。

    功能概述：
    - 对每条输出的 `<think>` 段进行长度度量（以词数计），按最大参考长度做 0~1 归一化，
      再线性映射到区间 `[length_min_possible, length_max_possible]` 作为奖励。

    动态上限：
    - 若设置环境变量 `SCHEDULELENGTH=1`，则最大参考长度 `max_reward_len` 会随 `global_step` 线性增长：
        max_reward_len = (640 - 384) * global_step / 105 + 384
      否则固定为 512。

    输入/输出：
    - 输入 `completions` 为模型输出文本列表，`solution` 仅作齐次遍历使用；
    - 返回与 `completions` 对齐的浮点奖励列表。
    """
    def __init__(self):
        """初始化长度奖励的默认上下界。"""
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        """
        计算每条输出的思考长度奖励。

        参数：
        - completions (List[str]): 模型输出文本列表。
        - solution (List[str]): 与 `completions` 对齐的标注列表（本函数不直接使用其内容，仅用于 zip 遍历）。
        - kwargs: 需包含键 `trainer_state`，以获取 `global_step`。

        环境变量：
        - SCHEDULELENGTH=1：启用动态最大参考长度，随 `global_step` 线性增长；否则固定为 512。

        返回：
        - List[float]: 归一化并线性映射后的奖励分数列表。
        """
        # 读取长度奖励可达的上界与下界（用于输出线性映射）
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        # 从 kwargs 读取训练状态并获取当前步数，用于动态上限
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            # 随训练步数线性增长最大参考长度（下限 384，上限趋近 640）
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            # 固定长度上限
            max_reward_len = 512
        # 逻辑说明：更长的 `<think>` 段在不超过上限时得到更高奖励
        responses = completions
        rewards = []  # 累积各样本奖励

        for response, ans in zip(responses, solution):
            # 若输出未包含 `<think>` 成对标签，则给最小奖励
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            # 提取 `<think>` 段内容并去除首尾空白
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            # 以空白分词统计词数，按最大参考长度归一化并保留两位小数
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            # 归一化得分不超过 1.0
            if reward > 1.0:
                reward = 1.0
            # 线性映射到 [min_possible_reward, max_possible_reward]
            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            # 记录该样本奖励
            rewards.append(final_reward)

        # 返回所有样本的奖励列表
        return rewards


class ToolUseCorrectnessReward(ORM):
    """
    工具调用正确性奖励函数。

    功能概述：
    - 比较 `<tool_call>` 段中预测的工具列表与标注的工具列表的一致性，并返回区间
      `[tool_min_possible, tool_max_possible]` 内的分数。支持粗粒度、阶段式与细粒度三种粒度控制：
        - COARSEREWARD=1：只要任一处不一致即取最小值；
        - INTERMEDIATEREWARD=1：仅在整条工具（含参数）完全一致时加分；
        - 默认（细粒度）：按工具名匹配度、参数名覆盖度、参数值正确性累计部分得分。

    动态范围：
    - 若 CORRECTMAX1=1：上下界为 [-1, 1]；否则为 [-3, 3]；
    - 若 MAX1STEP30MAX3=1：在 `global_step < 30` 阶段将区间等比缩小至原来的 1/3；
    - 若 SCHEDULEREWARD=1：按 `global_step` 对上下界进行线性插值，并裁剪至 [-3, 3]。

    输入/输出：
    - 输入 `completions` 为模型输出文本，`solution` 为带 `<tool_call>` 段的标注文本；
    - 输出与 `completions` 对齐的奖励分数列表。
    """
    def __init__(self):
        """
        初始化工具正确性奖励的动态上下界：
        - 当 `CORRECTMAX1=1`，区间为 [-1, 1]；否则区间为 [-3, 3]。
        """
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        """
        计算两个列表的匹配度分数，考虑元素频次的交并度量。

        规则：
        - 完全相等直接返回 1.0；
        - 若 `REFINEDREWARD=1`，且列表不完全相等，直接返回 0.0；
        - 否则使用带频次的交并度量：score = intersection / (len1 + len2 - intersection)。

        参数：
        - list1 (List[Any]): 参考列表；
        - list2 (List[Any]): 预测列表；

        返回：
        - float: 匹配度分数，范围 [0, 1]。
        """
        # 完全一致直接返回 1.0
        if list1 == list2:
            return 1.0

        # 细化奖励开关下，任何不一致直接 0.0
        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        # 任一为空，无法匹配
        if not list1 or not list2:
            return 0.0

        # 统计词频以进行带频次的交并度量（Jaccard 相似度）
        count1 = Counter(list1)  # list1 词频
        count2 = Counter(list2)  # list2 词频

        # 交集频次之和
        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        # 并集等效计数：len1 + len2 - intersection
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        """
        计算工具调用的细粒度奖励。

        规则：
        - 完全一致直接返回 `max_possible_reward`；
        - 若 `COARSEREWARD=1` 且任一处不一致，直接返回 `min_possible_reward`；
        - 否则：
            1) 先按工具名列表计算匹配得分；
            2) 对每个标注工具，在预测集中寻找同名的“最佳匹配”工具：
               - 若 `INTERMEDIATEREWARD=1`：只有完全相等才加 1.0；
               - 否则：按参数名覆盖度与参数值正确性累计部分分。
            3) 将累计分数按 `local_max_possible` 线性缩放到 `[min_possible_reward, max_possible_reward]`。

        参数：
        - gt_tools (List[Dict]): 标注工具列表，每项包含 `name` 与 `parameters`；
        - pd_tools (List[Dict]): 预测工具列表；
        - max_possible_reward (float): 当前阶段可达的最大奖励；
        - min_possible_reward (float): 当前阶段可达的最小奖励。

        返回：
        - float: 线性缩放后的细粒度奖励。
        """
        # 完全一致直接返回最大奖励
        if gt_tools == pd_tools:
            return max_possible_reward

        # 粗粒度模式：存在不一致即取最小值
        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        # 1) 工具名列表的匹配度
        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        # 后续可累计的局部最大分，将根据奖励粒度动态扩展
        local_max_possible = 1.0
        used_pd_indices = set()  # 记录已匹配的预测工具索引，避免重复配对

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            # 依据粒度调整局部上限：
            # - 中等粒度：仅考察是否完全匹配该条工具（+1.0）
            # - 细粒度：除工具名匹配外，参数名覆盖度与参数值正确性也纳入总分
            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            # 为当前 gt_tool 在未使用的 pd_tools 中寻找最佳匹配
            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                # 中等粒度：只有完全一致才加 1.0
                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                # 细粒度：参数名覆盖度 + 参数值正确性
                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # 按参数名覆盖度与参数值相等计分
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        # 线性缩放累积分数到 [min_possible_reward, max_possible_reward]
        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        """
        比对 `<tool_call>` 段内容，返回细粒度或粗粒度的正确性奖励。

        参数：
        - completions (List[str]): 模型输出文本列表。
        - solution (List[str]): 标注文本列表，需包含 `<tool_call>` 段以给出参考工具列表；
        - kwargs: 需包含键 `trainer_state`，其 `global_step` 用于动态调整奖励区间。

        环境变量：
        - MAX1STEP30MAX3=1：训练前 30 步缩放奖励区间至原来的 1/3；
        - SCHEDULEREWARD=1：对区间端点做线性插值并裁剪至 [-3, 3]；
        - COARSEREWARD / INTERMEDIATEREWARD / REFINEDREWARD：控制匹配粒度。

        返回：
        - List[float]: 按样本的正确性奖励分数。
        """
        # 读取训练步数以确定动态区间
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # 初始奖励区间为构造函数设定的上限与下限
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        # 在前 30 步缩小区间，后续恢复
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        # 对区间端点做线性插值，并裁剪到 [-3.0, 3.0]
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        # 为可读性重命名
        responses = completions
        rewards = []  # 累积各样本奖励

        for response, ans in zip(responses, solution):
            # 默认分数 0.0（若标注没有 `<tool_call>`，直接返回 0.0）
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            # 解析标注工具列表
            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                # 若预测文本缺失 `<tool_call>` 成对标签，直接给最小分
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                # 解析预测工具列表
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                # 计算细粒度/中等粒度/粗粒度奖励
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                # 解析失败或格式不合法，给最小分
                reward = min_possible_reward

            # 记录该样本得分
            rewards.append(reward)

        # 返回全部样本的奖励列表
        return rewards


orms['external_math_acc'] = MathAccuracy  # 数学答案符号等价奖励
orms['external_math_format'] = MathFormat  # 数学答案输出格式奖励
orms['external_countdown'] = CountdownORM  # 倒计时题格式与正确性奖励
orms['external_r1v_acc'] = MultiModalAccuracyORM  # 多模态答案准确性（符号验证+降级串匹配）
orms['external_code_reward'] = CodeReward  # E2B 沙箱执行奖励
orms['external_code_format'] = CodeFormat  # 代码格式合规奖励
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0  # Judge0 在线评测奖励
orms['external_tooluse_format_reward'] = ToolUseFormatReward  # 工具使用格式奖励
orms['external_tooluse_length_reward'] = ToolUseLengthReward  # 工具使用思考长度奖励
orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward  # 工具调用正确性奖励
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    自定义奖励模型插件（等价于 `DefaultRMPlugin` 的简化实现）。

    约定 `self.model` 为带 value head（输出维度为 1）的分类模型，
    其输出的第一列 logit 作为奖励分数。
    """

    def __init__(self, model, template):
        """保存模型与模板实例。"""
        # 记录用于打分的模型（应为带 value head 的分类模型，输出 logits[:, 0] 作为分数）
        self.model = model
        # 记录推理所需模板，负责编码与 batch 构造
        self.template: Template = template

    def __call__(self, inputs):
        """对一批输入推理并返回奖励张量（第一列 logit）。"""
        # 使用模板对每条推理请求进行编码，深拷贝确保不修改原始输入
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        # 通过模板的 data_collator 组 batch，并移动到模型设备上
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        # 关闭梯度以提升推理性能
        with torch.inference_mode():
            # 取分类模型的第一列 logit 作为奖励分数（shape: [batch]）
            return self.model(**reward_inputs).logits[:, 0]


class QwenLoingPlugin(DefaultRMPlugin):
    def __init__(self, model, template, accurancy_orm=None):
        super().__init__(model, template)
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)
        self.request_config = RequestConfig(temperature=0)
        self.system = textwrap.dedent('')


class QwenLongPlugin(DefaultRMPlugin):
    """
    QwenLong 风格的奖励模型插件。

    使用一个判定提示（system prompt）比较“回答1（模型输出）”与“回答2（标准答案）”是否等价；
    结合外部 `accuracy_orm` 的符号/串匹配校验，最终取两者较大值作为奖励。
    """
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        """初始化推理引擎、请求配置与系统提示，并保存外部验证器。"""
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        """主入口：构建 RM 输入、调用推理并融合多源奖励。"""
        # 从输入样本中抽取模型输出文本与对应的标准答案
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        # 构建适配奖励模型的单轮对话输入
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        # 调用推理引擎进行评分推理
        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        # 将模型输出转为 0/1 并取平均，得到 LLM 评分
        llm_rewards = self.compute_rewards(results)

        # 若提供外部准确性验证器（如符号验证/字符串比对），取两者中的较大值
        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        # 融合奖励：逐样本取最大值，避免 LLM 评分与外部验证冲突时过度惩罚
        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        # 返回浮点张量
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        """
        将原始对话转为奖励模型的单轮输入：
        - 从首轮消息中提取问题文本；
        - 用预设 `self.system` 模板填充问题与两个答案；
        - 保持其它推理配置不变。
        """
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # 深拷贝输入，避免修改原始推理请求
            rm_infer_request = deepcopy(infer_request)
            # 取第一轮消息作为问题文本，按模板约定切片出实际问题
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            # 将问题与两份答案填入系统提示模板，构造用于判等的 prompt
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # 组装新的单轮消息，供奖励模型评分
            rm_messages = [{'role': 'user', 'content': prompt}]

            # 写回到新的推理请求对象
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        """从模型输出中抽取 [[YES]]/[[NO]] 并转为 1.0/0.0 奖励。"""
        # 以正则匹配形如 [[YES]] 或 [[NO]] 的结论标记
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            # 读取捕获的标记
            answer = match.group(1)
            # YES -> 1.0
            if answer == 'YES':
                return 1.0
            # NO -> 0.0
            elif answer == 'NO':
                return 0.0
            else:
                # 其他标记记为 0.0 并给出警告
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            # 未能匹配结论标记，统一返回 0.0
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        从奖励模型输出中计算平均奖励：遍历候选，转换为 0/1 并取平均。

        参数:
            results (List[ChatCompletionResponse]): 奖励模型的推理结果列表。

        返回:
            List[float]: 平均奖励分数。
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                # 遍历该样本的所有候选输出
                for choice in output.choices:
                    response = choice.message.content
                    # 抽取 [[YES]]/[[NO]] 并映射为 1.0/0.0
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                # 过滤掉 None（理论上不会有 None，这里为稳健性保留）
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    # 多候选取平均分
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                # 捕获异常并为该样本返回 0.0，避免整体失败
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin  # 简化 RM 插件（value head logit）
rm_plugins['qwenlong'] = QwenLongPlugin  # QwenLong 风格 RM（LLM 判定 + 外部校验）
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step() (Required): Constructs the next round of the infer request.
            - check_finished() (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
        Both methods accept
            - the last turn's InferRequest/result
            The current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --multi_turn_scheduler my_scheduler
"""


class ReToolScheduler(MultiTurnScheduler):
    """
    多轮对话调度器示例。

    继承自基类 `MultiTurnScheduler`，此处占位实现（直接复用默认行为）。
    可自定义 `step`/`check_finished` 以实现特定回合逻辑。
    """
    pass


multi_turns['retool'] = ReToolScheduler  # 多轮调度器注册


# register GYM env
class CustomEnv(Env):
    """
    自定义 Gym 环境占位类。

    继承自 `Env`，当前未添加新行为，仅用于注册与示例扩展。
    """
    pass


envs['custom_env'] = CustomEnv  # Gym 环境注册


class CustomCtxManager(ContextManager):
    """
    自定义上下文管理器占位类。

    可在 Gym 训练或推理期间注入/清理上下文资源。
    """
    pass


context_managers['custom_ctx'] = CustomCtxManager  # 上下文管理器注册
