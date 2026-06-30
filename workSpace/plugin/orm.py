import os
import re
from typing import Dict, List, Union, Optional

import json
from transformers import AutoTokenizer
from swift.llm import InferRequest


class ORM:
    """
    奖励模型基类 (Outcome Reward Model)
    
    用于强化学习中评估模型输出质量的基础接口。
    所有具体的奖励模型都应该继承此类并实现 __call__ 方法。
    """

    def __call__(self, **kwargs) -> List[float]:
        """
        计算奖励值的抽象方法
        
        Args:
            **kwargs: 可变参数，具体参数由子类定义
            
        Returns:
            List[float]: 每个样本对应的奖励值列表
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


class ReactORM(ORM):
    """
    ReAct (Reasoning and Acting) 框架的奖励模型
    
    用于评估基于 ReAct 框架的智能体输出质量。
    ReAct 框架结合了推理（Reasoning）和行动（Acting），
    通过思考-行动-观察的循环来解决问题。
    """

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        """
        评估动作执行的奖励值
        
        通过比较预测的动作和参考动作来计算奖励。
        支持两种输入格式：JSON格式和纯文本格式。
        
        Args:
            action_pred (list): 预测的动作列表
            action_ref (list): 参考（标准）动作列表  
            cand_list (list): 候选动作输入列表
            ref_list (list): 参考动作输入列表
            
        Returns:
            bool: 如果第一个样本的F1分数为1.0则返回True，否则返回False
        """
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            # 尝试解析参考输入为JSON格式
            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            # 尝试解析候选输入为JSON格式
            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            # 如果动作不匹配或JSON格式不一致，奖励为0
            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            # 如果都不是JSON格式，使用ROUGE-L评估文本相似度
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)  # 部分匹配
                else:
                    f1.append(1)    # 完全匹配
            # 如果都是JSON格式，比较字典内容
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # 防御性编程：处理非字典类型的边缘情况
                    f1.append(0)
                    continue

                half_match = 0  # 部分匹配的键值对数量
                full_match = 0  # 完全匹配的键值对数量
                
                # 处理空字典的情况
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    # 逐个比较键值对
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    # 计算召回率和精确率
                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    
                    # 计算F1分数
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        # 返回第一个样本是否完全正确
        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        """
        从文本中解析动作和动作输入
        
        解析 ReAct 框架中的标准格式：
        Action: <动作名称>
        Action Input: <动作输入>
        
        Args:
            text (str): 要解析的文本
            
        Returns:
            tuple: (动作名称, 动作输入)
        """
        # 解析动作输入
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')  # 从右侧查找最后一个匹配项
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'  # 默认空输入

        # 解析动作名称
        if 'Action:' in text:
            action_idx = text.rindex('Action:')  # 从右侧查找最后一个匹配项
            action = text[action_idx + len('Action:'):].strip()
            # 如果动作文本中包含Action Input，则截取到Action Input之前
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'  # 默认无动作
            
        return action, action_input

    @staticmethod
    def parse_output(text):
        """
        解析输出文本，提取动作和动作输入
        
        这是 parse_action 的包装方法，用于保持接口一致性
        
        Args:
            text (str): 要解析的输出文本
            
        Returns:
            tuple: (动作名称, 动作输入)
        """
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], solution: List[str], **kwargs) -> List[float]:
        """
        计算 ReAct 任务的奖励值
        
        Args:
            infer_requests: 推理请求列表，可以是InferRequest对象或字典
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的奖励值列表（0.0或1.0）
        """
        rewards = []
        
        # 提取预测结果
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
            
        # 逐个比较预测和标准答案
        for prediction, ground_truth in zip(predictions, solution):
            # 移除末尾的观察标记
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
                
            # 初始化动作和输入列表
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            
            reference = ground_truth
            # 清理预测文本中的特殊标记
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            
            # 解析参考答案和预测结果
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            
            # 添加到列表中
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            
            # 处理预测结果的空值情况
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            # 计算奖励
            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
            
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        """
        使用ROUGE-L评估文本相似度
        
        ROUGE-L基于最长公共子序列（LCS）来评估文本质量。
        
        Args:
            cand_list (list): 候选文本列表
            ref_list (list): 参考文本列表
            
        Returns:
            float or None: ROUGE-L F1分数，如果计算失败则返回None
        """
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']  # 提取ROUGE-L的F1分数
            return rougel
        except Exception:
            return None


class MathORM(ORM):
    """
    数学问题奖励模型
    
    专门用于评估数学问题求解的正确性。
    支持LaTeX格式的数学表达式比较。
    """

    def __init__(self):
        """
        初始化数学奖励模型
        
        根据环境变量决定是否使用OpenCompass评估器
        """
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        """
        检查答案是否包含终止标记
        
        在数学问题中，通常使用 \\boxed{} 来标记最终答案
        
        Args:
            answers: 单个答案字符串或答案列表
            
        Returns:
            List[bool]: 每个答案是否包含终止标记的布尔值列表
        """
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        """
        从文本中提取被 \\boxed{} 包围的内容
        
        Args:
            text (str): 包含数学表达式的文本
            
        Returns:
            str: 提取的内容，如果没有找到则返回原文本
        """
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        """
        清理LaTeX字符串
        
        移除常见的LaTeX标记，如 \\(, \\), \\[, \\], {, }
        
        Args:
            latex_str (str): 原始LaTeX字符串
            
        Returns:
            str: 清理后的字符串
        """
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        """
        解析LaTeX数学表达式
        
        使用sympy库将LaTeX字符串转换为数学表达式并简化
        
        Args:
            latex_str (str): LaTeX格式的数学表达式
            
        Returns:
            sympy.Basic or None: 解析后的数学表达式，失败时返回None
        """
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        """
        比较两个数学表达式是否相等
        
        先清理LaTeX格式，然后解析为数学表达式进行比较
        
        Args:
            first (str): 第一个数学表达式
            second (str): 第二个数学表达式
            
        Returns:
            bool: 两个表达式是否数学上相等
        """
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        
        # 使用sympy的equals方法进行数学等价性比较
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
            
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        """
        计算数学问题的奖励值
        
        Args:
            infer_requests: 推理请求列表
            ground_truths: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的奖励值列表（0.0或1.0）
        """
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        
        for prediction, ground_truth in zip(predictions, ground_truths):
            # 提取答案部分（如果有# Answer标记）
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
                
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            
            # 提取boxed内容
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            
            # 使用相应的评估器进行比较
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
                
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):
    """
    数学准确性奖励模型
    
    使用 math_verify 包来验证数学答案的正确性。
    这是另一种数学答案验证方法，提供更严格的验证。
    """

    def __init__(self):
        """
        初始化数学准确性模型
        
        检查 math_verify 包是否已安装
        """
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        使用math_verify包验证数学答案
        
        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的奖励值列表（0.0或1.0）
        """
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        
        rewards = []
        for content, sol in zip(completions, solution):
            # 解析标准答案
            gold_parsed = parse(sol, extraction_mode='first_match')
            
            if len(gold_parsed) != 0:
                # 解析模型答案，要求严格的LaTeX格式
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,  # 不允许格式错误的操作符
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            boxed_match_priority=0,  # 优先匹配boxed格式
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                
                # 验证答案正确性
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # 如果标准答案无法解析，则跳过此样本
                reward = 0.0
                
            rewards.append(reward)
        return rewards


class Format(ORM):
    """
    格式检查奖励模型
    
    检查模型输出是否符合特定的格式要求。
    要求输出包含 <think>...</think> 和 <answer>...</answer> 标签。
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查输出格式是否正确
        
        期望的格式：<think>思考过程</think> <answer>最终答案</answer>
        
        Args:
            completions: 模型输出列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的奖励值（1.0表示格式正确，0.0表示格式错误）
        """
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):
    """
    ReAct格式检查奖励模型
    
    检查输出是否符合ReAct框架的格式要求。
    要求包含思考过程、动作和动作输入。
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查ReAct格式是否正确
        
        期望的格式：<think>思考过程</think> Action: 动作名称 Action Input: 动作输入
        
        Args:
            completions: 模型输出列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的奖励值（1.0表示格式正确，0.0表示格式错误）
        """
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    """
    余弦奖励模型
    
    基于输出长度的余弦函数奖励机制。
    参考论文: https://arxiv.org/abs/2502.03373
    
    对于正确答案，倾向于奖励较短的输出；
    对于错误答案，则相反。
    """
    
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        """
        初始化余弦奖励模型
        
        Args:
            tokenizer: 用于计算token长度的分词器
            cosine_min_len_value_wrong: 错误答案最短长度时的奖励值
            cosine_max_len_value_wrong: 错误答案最长长度时的奖励值
            cosine_min_len_value_correct: 正确答案最短长度时的奖励值
            cosine_max_len_value_correct: 正确答案最长长度时的奖励值
            cosine_max_len: 最大长度阈值
            accuracy_orm: 用于判断答案正确性的奖励模型
        """
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        """
        余弦函数计算奖励值
        
        Args:
            t: 当前长度
            T: 最大长度
            min_value: 最小奖励值
            max_value: 最大奖励值
            
        Returns:
            float: 计算得到的奖励值
        """
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        基于长度和正确性计算余弦奖励
        
        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的余弦奖励值
        """
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            
            # 根据答案正确性选择奖励参数
            if is_correct:
                # 对于正确答案，短的更好（交换min/max）
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
                
            # 计算生成长度并应用余弦函数
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
            
        return rewards


class RepetitionPenalty(ORM):
    """
    重复惩罚奖励模型
    
    通过检测n-gram重复来惩罚重复性内容。
    参考论文: https://arxiv.org/abs/2502.03373
    """
    
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        """
        初始化重复惩罚模型
        
        Args:
            repetition_n_grams: n-gram的大小
            repetition_max_penalty: 最大惩罚值
        """
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        """
        生成文本的n-gram序列
        
        Args:
            text: 输入文本
            ngram_size: n-gram大小
            
        Returns:
            generator: n-gram元组的生成器
        """
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        计算重复惩罚奖励
        
        通过计算唯一n-gram与总n-gram的比例来评估重复度
        
        Args:
            completions: 模型输出列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的重复惩罚值（负值表示惩罚）
        """
        rewards = []
        for completion in completions:
            # 处理空输出
            if completion == '':
                rewards.append(0.0)
                continue
                
            # 处理过短的输出
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            # 计算唯一n-gram数量
            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            # 计算重复度并应用惩罚
            scaling = 1 - len(ngrams) / total  # 重复度（0-1，1表示完全重复）
            reward = scaling * self.max_penalty  # 应用惩罚
            rewards.append(reward)
            
        return rewards


class SoftOverlong(ORM):
    """
    软长度限制奖励模型
    
    对超过期望长度的输出进行软惩罚，而不是硬截断。
    """

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        """
        初始化软长度限制模型
        
        Args:
            tokenizer: 分词器
            soft_max_length: 软最大长度
            soft_cache_length: 软缓存长度（必须小于soft_max_length）
        """
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        计算长度惩罚奖励
        
        对超过期望长度的部分进行线性惩罚
        
        Args:
            completions: 模型输出列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的长度惩罚值（非正值）
        """
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            
            # 计算超长惩罚：超出部分按比例惩罚，最大惩罚为0
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
            
        return rewards


##-------------
tool_list = json.load(open('/mnt/cfs/ssw/wx/code/interface/ssw_chat_agent/configs/tools_list.json', 'r', encoding='utf-8'))
tool_list_info = {}
for item in tool_list:
    tool_list_info[item['function']['name']]=item['function']['parameters']

class AgentAccReward(ORM):
    
    def __init__(self):
        from bs4 import BeautifulSoup
        from swift.plugin.card_validate import CardValidator
        
        self.validator = CardValidator()
        
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查模型回复的内容是否正确
        
        Args:
            completions: 模型回复的消息列表，每个元素是一个消息列表
            global_step: 当前训练步数（用于动态奖励调度）
            **kwargs: 其他参数
        
        Returns:
            List[float]: 每个样本的奖励值列表
        """
        rewards = []
        
        for completion in completions:
            try:
                reward = self._evaluate_single_completion(
                    completion
                )
                rewards.append(reward)
            except Exception as e:
                print(f"评估过程中出现错误: {e}")
                
        return rewards

    def _evaluate_single_completion(self, messages: List[Dict]) -> bool:
        """
        评估单个completion的奖励分数
        
        Args:
            messages: 单个completion的消息列表
 
        Returns:
            bool: 该completion的奖励分数
        """
        # 初始化奖励组件
        format_reward_score = 0.0
        planner_reward_score = 0.0
        
        # 解析消息内容
        tool_name = None
        tool_args = None
        tool_content = None
        output_content = None
        
        # 遍历消息提取关键信息
        for message in messages:
            if message['role'] == 'assistant' and 'tool_calls' in message:
                if message['tool_calls']:
                    tool_name = message['tool_calls'][0]['function']['name']
                    tool_args = message['tool_calls'][0]['function']['arguments']
                    
            elif message['role'] == 'tool':
                raw_content = message['content']
                # 标准化工具返回内容
                tool_content = self._load_tool_content(raw_content)
                
            elif message['role'] == 'assistant' and 'content' in message:
                output_content = message.get('content')
        
        # 检查基本格式
        if not output_content:
            return False
            
        # 验证输出内容
        self.validator.check_card_by_type(tool_name, output_content, tool_content)
        
        # 如果有工具调用
        if tool_name:
            # 检查工具参数
            tool_args_valid, tool_args_error = self.check_tool_args(tool_name, tool_args)
            
            # 参数奖励计算
            planner_reward_score += 1.0 if tool_args_valid else -1.0
            
            # 根据工具返回内容和验证结果计算奖励
            if tool_content is not None:  # 有工具返回
                if self.validator.validation_result['output_type'] == 'ssw-card':
                    # 返回了ssw-card类型
                    if self.validator.validation_result['valid'] and tool_args_valid:
                        return True  # 最佳情况：卡片有效且参数正确
                    elif not tool_args_valid:
                        return False  # 参数错误直接最低分
                    else:
                        return False  # 卡片无效但参数正确
                else:
                    # 返回了普通文本
                    if self.validator.validation_result['valid']:
                        return True
                    else:
                        if not tool_args_valid:
                            return False    
            else:  # 无工具返回
                if self.validator.validation_result['output_type'] == 'ssw-card':
                    return False  # 应该有工具返回但没有，且输出了卡片
                else:
                    # 正确处理了无返回情况
                    if self.validator.validation_result['valid'] and tool_args_valid:
                        return True
                    else:
                        if not tool_args_valid:
                            return False
        else:  # 无工具调用
            if self.validator.validation_result['output_type'] == 'ssw-card':
                # 不应该输出卡片但输出了
                return False
            else:
                # 纯文本回答
                if self.validator.validation_result['valid']:
                    return True  # 正确的文本回答
                else:
                    return False  # 错误的文本回答
        
        return False
    
    def _load_tool_content(self, raw_content: str) -> Optional[Dict]:
        """
        标准化工具返回内容
        
        Args:
            raw_content: 原始工具返回内容
            
        Returns:
            Optional[Dict]: 解析后的内容，如果无效则返回None
        """
        if not raw_content or raw_content in ['未搜索到相关内容', '没有返回内容', '']:
            return None
            
        try:
            return json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def check_tool_args(tool_call_name: str, tool_args: str) -> tuple[bool, Optional[str]]:
        """检查工具参数"""
        if not tool_call_name or tool_call_name not in tool_list_info:
            return False, f"工具{tool_call_name}不存在"
            
        try:
            type_map = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'array': list, 'object': dict}
            parsed_args = json.loads(tool_args)
            args_define = tool_list_info[tool_call_name]['properties']
            
            for arg_name, arg_value in parsed_args.items():
                if arg_name not in args_define:
                    return False, f"工具参数{arg_name}不在工具定义中"
                    
                expected_type = type_map.get(args_define[arg_name]['type'])
                if expected_type and not isinstance(arg_value, expected_type):
                    return False, f"工具参数{arg_name}类型不匹配"
                    
                if 'enum' in args_define[arg_name]:
                    if arg_value not in args_define[arg_name]['enum']:
                        return False, f"工具参数{arg_name}枚举值不匹配"
            
            # 检查required参数是否存在
            required_params = tool_list_info[tool_call_name].get('required', [])
            for required_param in required_params:
                if required_param not in parsed_args:
                    return False, f"必需参数{required_param}不存在"
                    
            return True, None
            
        except json.JSONDecodeError:
            return False, "工具参数JSON格式错误"
        except Exception as e:
            return False, f"参数检查出错: {str(e)}"


class CombinedCosineReward(ORM):
    """
    余弦奖励模型
    
    基于输出长度的余弦函数奖励机制。
    参考论文: https://arxiv.org/abs/2502.03373
    
    对于正确答案，倾向于奖励较短的输出；
    对于错误答案，则相反。
    """
    
    def __init__(self,
                 tokenizer,
                 cosine_value_wrong: float = -1.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        """
        初始化余弦奖励模型
        
        Args:
            tokenizer: 用于计算token长度的分词器
            cosine_min_len_value_wrong: 错误答案最短长度时的奖励值
            cosine_max_len_value_wrong: 错误答案最长长度时的奖励值
            cosine_min_len_value_correct: 正确答案最短长度时的奖励值
            cosine_max_len_value_correct: 正确答案最长长度时的奖励值
            cosine_max_len: 最大长度阈值
            accuracy_orm: 用于判断答案正确性的奖励模型
        """
        self.tokenizer = tokenizer
        self.cosine_value_wrong = cosine_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = AgentAccReward()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        """
        余弦函数计算奖励值
        """
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        基于长度和正确性计算余弦奖励
        
        Args:
            completions: 模型输出列表
            solution: 标准答案列表
            **kwargs: 其他参数
            
        Returns:
            List[float]: 每个样本的余弦奖励值
        """
        acc_rewards = self.accuracy_orm(completions, **kwargs)
        rewards = []
        
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            
            # 根据答案正确性选择奖励参数
            if is_correct:
                # 对于正确答案，短的更好（交换min/max）
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                reward = self.cosine_value_wrong
                
            # 计算生成长度并应用余弦函数
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
            
        return rewards



# 奖励模型注册字典
# 将奖励模型名称映射到对应的类，便于动态加载
orms = {
    'toolbench': ReactORM,      # 工具台/ReAct任务奖励模型
    'math': MathORM,            # 数学问题奖励模型
    'accuracy': MathAccuracy,   # 数学准确性奖励模型
    'format': Format,           # 格式检查奖励模型
    'react_format': ReActFormat, # ReAct格式检查奖励模型
    'cosine': CosineReward,     # 余弦奖励模型
    'repetition': RepetitionPenalty,  # 重复惩罚奖励模型
    'soft_overlong': SoftOverlong,    # 软长度限制奖励模型
    'agent_acc': AgentAccReward,     # 工具调用 正误判断
    'combined_cosine': CombinedCosineReward    # 余弦奖励模型+工具调用正误判断
}
