import re, json
from typing import Optional, Dict
from typing import List
from transformers import AutoTokenizer
import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建一个 StreamHandler，默认输出到 stderr
handler = logging.StreamHandler(sys.stdout)  # 如果你希望输出到 stdout，也可以保留默认而不传 sys.stdout
handler.setLevel(logging.DEBUG)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# 添加 handler 到 logger
logger.addHandler(handler)

tool_list = json.load(open('/mnt/cfs/ssw/wx/code/interface/ssw_chat_agent/configs/tools_list.json', 'r', encoding='utf-8'))
tool_list_info = {}
for item in tool_list:
    tool_list_info[item['function']['name']]=item['function']['parameters']

def _get_tokenizer():
    """Get tokenizer instance with thread safety"""
    global _tokenizer
    _tokenizer = None
    if _tokenizer is None:
        tokenizer_dir = "/mnt/cfs/ssw/ljc/gits/chat_conversation/data/Qwen3-235B-A22B-FP8"
        try:
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        except Exception as e:
            print(f"Error loading tokenizer from {tokenizer_dir}: {e}")
            _tokenizer = None
    return _tokenizer

tokenizer = _get_tokenizer()

class NotToolCallAccReward():
    def __init__(self):
        import sys
        import os
        self.min_score = 0
        self.max_score = 1
        
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查模型回复的内容是否正确
        
        Args:
            completions: 模型回复的消息列表，每个元素是一个消息列表
                each for completions example: 
                    <think>\n好的，用户让我翻译“wednesday”，我需要调用查询英语单词的工具。首先，确认工具名称是query_en_word，参数是word，所以调用这个工具，传入“wednesday”作为参数。工具返回的结果里有很多条目，每个条目都有不同的年级和版本，但翻译都是“星期三”。需要检 查是否有错误或需要特别注意的地方。工具返回的数据结构正确，没有问题。接下来根据规则，不需要使用box模块，因为这是查询单词的工具，直接返回翻译即可。所以回复用户“Wednesday”的翻译是“星期三”。\n</think>\n\n<ssw-card>\n    <print>Wednesday的翻译是：星期三</print>\n</ssw-card>

            global_step: 当前训练步数（用于动态奖励调度）
            **kwargs: 其他参数
        
        Returns:
            List[float]: 每个样本的奖励值列表
        """
        rewards = []
        tc_val_list = []
        # N_count = 0
        # logger.info(f"rollout completions: {[completion for completion in completions]}")
        for completion in completions:
            try:
                reward, should_do_tc = self._evaluate_single_completion(completion)
                rewards.append(reward)
                tc_val_list.append(should_do_tc)
                # if should_do_tc == "N":
                    # N_count += 1
            except Exception as e:
                print(f"评估过程中出现错误: {e}")
                rewards.append(0.0)  # 错误时给予最低奖励

        group_rewards = []
        # 如果有成功，即认为tc为true
        if "Y" in tc_val_list:
            for reward, should_do_tc in zip(rewards, tc_val_list):
                if should_do_tc == "N":
                    group_rewards.append(0)
                else:
                    group_rewards.append(reward)
        else:
            group_rewards = rewards
        # 投票方法
        # if count <= len(completions) // 2:
        #     for reward, should_do_tc in zip(rewards, tc_val_list):
        #         if should_do_tc == "N":
        #             group_rewards.append(0)
        #         else:
        #             group_rewards.append(reward)
                
        return group_rewards

    def _evaluate_single_completion(self, completion):
        """
        评估单个completion的奖励分数
        
        Args:
            completion: 单个content,格式：<think>...</think>...
        Returns:
            bool: 该completion的奖励分数
        """
        
        # 解析消息内容
        tool_name = None
        tool_args = None
        tool_content = None
        # logger.info(f"completion: {completion}")

        # TODO: 验证rollout生成时的 tc 格式
        think_pattern = re.compile(r'<think>(.*?)</think>\n\n(.*)', re.DOTALL)
        think_match = think_pattern.search(completion)
        if think_match:
            think_content = think_match.group(1)
            not_think_content = think_match.group(2)
            # logger.info(f"think_content: {think_content}")
            # logger.info(f"not_think_content: {not_think_content}")
        else:
            logger.info(f"<think> tag compile not match")
            return self.min_score, "N"

        if 'ssw-card' in not_think_content:
            return self.min_score, "N"

        check_words = ['function', 'name', 'arguments']
        if all(word in not_think_content for word in check_words):
            try:
                tool_content = json.loads(not_think_content)
                tool_name = tool_content['function']['name']
                tool_args = tool_content['function']['arguments']
                tool_args_valid, tool_args_error = self.check_tool_args(tool_name or "", tool_args or "")
                if not tool_args_valid:
                    return self.min_score, "F"
                return self.max_score, "Y"
            except json.JSONDecodeError:
                return self.min_score, "F"
        else:
            return self.max_score, "N"

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
    def check_tool_args(tool_call_name: str, tool_args: str) -> tuple:
        """检查工具参数"""
        logger.info(f"tool_list_info: {tool_list_info}")
        if not tool_call_name or tool_call_name not in tool_list_info:
            return False, f"工具{tool_call_name}不存在"
            
        try:
            type_map = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'array': list, 'object': dict}
            # parsed_args = json.loads(tool_args)
            parsed_args = tool_args
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


class ToolCallAccReward():
    def __init__(self):
        import sys
        import os
        self.suc_score = 1
        self.fail_score = 0
        self.ntc_score = -1
        
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        检查模型回复的内容是否正确
        
        Args:
            completions: 模型回复的消息列表，每个元素是一个消息列表
                each for completions example: 
                    <think>\n好的，用户让我翻译“wednesday”，我需要调用查询英语单词的工具。首先，确认工具名称是query_en_word，参数是word，所以调用这个工具，传入“wednesday”作为参数。工具返回的结果里有很多条目，每个条目都有不同的年级和版本，但翻译都是“星期三”。需要检 查是否有错误或需要特别注意的地方。工具返回的数据结构正确，没有问题。接下来根据规则，不需要使用box模块，因为这是查询单词的工具，直接返回翻译即可。所以回复用户“Wednesday”的翻译是“星期三”。\n</think>\n\n<ssw-card>\n    <print>Wednesday的翻译是：星期三</print>\n</ssw-card>

            global_step: 当前训练步数（用于动态奖励调度）
            **kwargs: 其他参数
        
        Returns:
            List[float]: 每个样本的奖励值列表
        """
        rewards = []
        # logger.info(f"rollout completions: {[completion for completion in completions]}")
        for completion in completions:
            try:
                reward, should_do_tc = self._evaluate_single_completion(completion)
                rewards.append(reward)
                logger.info(f"tool call status: {should_do_tc}")
            except Exception as e:
                print(f"评估过程中出现错误: {e}")
                rewards.append(0.0)  # 错误时给予最低奖励
        logger.info(f"rewards count: {len(rewards)}")
        return rewards

    def _evaluate_single_completion(self, completion):
        """
        评估单个completion的奖励分数
        
        Args:
            completion: 单个content,格式：<think>...</think>...
        Returns:
            bool: 该completion的奖励分数
        """
        
        # 解析消息内容
        tool_name = None
        tool_args = None
        tool_content = None
        # logger.info(f"completion: {completion}")

        # TODO: 验证rollout生成时的 tc 格式
        think_pattern = re.compile(r'<think>(.*?)</think>\n\n(.*)', re.DOTALL)
        think_match = think_pattern.search(completion)
        if think_match:
            think_content = think_match.group(1)
            not_think_content = think_match.group(2)
            # logger.info(f"think_content: {think_content}")
            # logger.info(f"not_think_content: {not_think_content}")
        else:
            logger.info(f"<think> tag compile not match")
            return self.ntc_score, "Thing Tag Error"

        if 'ssw-card' in not_think_content:
            return self.ntc_score, "SSW Card Error"

        check_words = ['function', 'name', 'arguments']
        if all(word in not_think_content for word in check_words):
            try:
                tool_content = json.loads(not_think_content)
                tool_name = tool_content['function']['name']
                tool_args = tool_content['function']['arguments']
                tool_args_valid, tool_args_error = self.check_tool_args(tool_name or "", tool_args or "")
                if not tool_args_valid:
                    return self.fail_score, "Failed"
                return self.suc_score, "Success"
            except json.JSONDecodeError:
                return self.ntc_score, "TC Josn Error"
        else:
            return self.ntc_score, "Not TC"

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
    def check_tool_args(tool_call_name: str, tool_args: str) -> tuple:
        """检查工具参数"""
        # logger.info(f"tool_list_info: {tool_list_info}")
        if not tool_call_name or tool_call_name not in tool_list_info:
            logger.info(f"工具{tool_call_name}不存在")
            return False, f"工具{tool_call_name}不存在"
            
        try:
            type_map = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'array': list, 'object': dict}
            # parsed_args = json.loads(tool_args)
            parsed_args = tool_args
            args_define = tool_list_info[tool_call_name]['properties']
            
            for arg_name, arg_value in parsed_args.items():
                if arg_name not in args_define:
                    logger.info(f"工具参数{arg_name}不在工具定义中")
                    return False, f"工具参数{arg_name}不在工具定义中"
                    
                expected_type = type_map.get(args_define[arg_name]['type'])
                if expected_type and not isinstance(arg_value, expected_type):
                    logger.info(f"工具参数{arg_name}类型不匹配")
                    return False, f"工具参数{arg_name}类型不匹配"
                    
                if 'enum' in args_define[arg_name]:
                    if arg_value not in args_define[arg_name]['enum']:
                        logger.info(f"工具参数{arg_name}枚举值不匹配")
                        return False, f"工具参数{arg_name}枚举值不匹配"
            
            # 检查required参数是否存在
            required_params = tool_list_info[tool_call_name].get('required', [])
            for required_param in required_params:
                if required_param not in parsed_args:
                    logger.info(f"必需参数{required_param}不存在")
                    return False, f"必需参数{required_param}不存在"
                    
            return True, None
            
        except json.JSONDecodeError:
            logger.info(f"工具参数JSON格式错误")
            return False, "工具参数JSON格式错误"
        except Exception as e:
            logger.info(f"参数检查出错: {str(e)}")
            return False, f"参数检查出错: {str(e)}"


class ToolCallsCosineReward():
    """
    余弦奖励模型
    
    基于输出长度的余弦函数奖励机制。
    参考论文: https://arxiv.org/abs/2502.03373
    
    对于正确答案，倾向于奖励较短的输出；
    对于错误答案，则相反。
    """
    
    def __init__(self,
                 tokenizer=tokenizer,
                 cosine_value_wrong: float = -1.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len_value_tc: float = 0,
                 max_len: int = 250,
                 accuracy_orm=ToolCallAccReward):
        """
        初始化余弦奖励模型
        
        Args:
            tokenizer: 用于计算token长度的分词器
            cosine_min_len_value_wrong: 错误答案最短长度时的奖励值
            cosine_max_len_value_wrong: 错误答案最长长度时的奖励值
            cosine_min_len_value_correct: 正确答案最短长度时的奖励值
            cosine_max_len_value_correct: 正确答案最长长度时的奖励值
            cosine_max_len: 最大长度阈值, 目前定为{tool_call时的平均长度 - 100}
            accuracy_orm: 用于判断答案正确性的奖励模型类
        """
        self.tokenizer = tokenizer
        self.cosine_value_wrong = cosine_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.min_len_value_tc = cosine_max_len_value_correct
        self.max_len_value_tc = cosine_max_len_value_tc
        self.max_len = max_len

        # 实例化accuracy_orm类
        self.accuracy_orm = accuracy_orm()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        """
        分段奖励函数：
        - 在T以内：保持最高奖励值
        - 超过T：指数衰减惩罚，在约T+200处趋近最低奖励值
        """
        import math
        
        # T以内保持最高奖励值
        # logger.info(f't: {t}, T: {T}, min_value: {min_value}, max_value: {max_value}')
        if t <= T:
            return max_value
        
        # 超过T时，使用指数衰减
        # 计算超出的长度
        excess_len = t - T
        logger.info(f'excess_len: {excess_len}')
        # 设置衰减参数，使得在200个token左右趋近min_value
        # 使用指数函数: reward = min_value + (max_value - min_value) * exp(-k * excess_len)
        # 当excess_len=200时，希望趋近min_value，设k使得exp(-k*200) ≈ 0.05
        k = 0.015  # 调整这个参数可以控制衰减速度
        
        decay_factor = math.exp(-k * excess_len)
        reward = min_value + (max_value - min_value) * decay_factor
        
        return reward

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
        logger.info(f"completions count: {len(completions)}")
        acc_rewards = self.accuracy_orm(completions, **kwargs)
        rewards = []
        
        for completion, acc_reward in zip(completions, acc_rewards):
            # 获取think内容和非think内容
            pattern = re.compile(r'<think>(.*?)</think>\n\n(.*?)', re.DOTALL)
            match = pattern.search(completion)
            if match:
                think_content = match.group(1)
                not_think_content = match.group(2)
            else:
                reward = self.cosine_value_wrong
                rewards.append(reward)
                continue
            
            if_correct = acc_reward == 1
            only_does_tc = acc_reward == 0
            
            # 根据答案正确性选择奖励参数
            if if_correct:
                # 对于正确答案，短的更好（交换min/max）
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
                # 计算生成长度并应用余弦函数
                gen_len = len(self.tokenizer.encode(think_content, add_special_tokens=True))
                logger.info(f'generated length: {gen_len}')
                reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            elif only_does_tc:
                min_value = self.max_len_value_tc
                max_value = self.min_len_value_tc
                # 计算生成长度并应用余弦函数
                gen_len = len(self.tokenizer.encode(think_content, add_special_tokens=True))
                logger.info(f'generated length: {gen_len}')
                reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            else:
                reward = self.cosine_value_wrong

            logger.info(f'reward: {reward}')
            rewards.append(reward)
            
        return rewards
        

if __name__ == "__main__":
    completion1 = """<think>
        好的，用户发来了"D I s c o v E r."，看起来像是打错了或者想测试我的反应。首先，我需要确认用户的真实意图。可能的情况是用户想表达"Discover"这个词，或者是在玩文字游戏，比如拆分字母或者隐藏的信息。

        接下来，我应该检查是否有需要调用的工具。用户的问题比较模糊，没有明确的请求，比如提问、生成题目或者查找资料。根据之前的指导，如果没有明确的需求，就不应该调用工具。此外，用户可能只是随意输入，或者想看看我会如何回应。

        考虑到用户可能希望得到某种互动或帮助，我需要用友好的语气回应，同时保持专业。根据规则，不能引导用户使用外部资源，也不能包含无关内容。因此，我应该用简单明了的话回应，邀请用户提供更具体的问题或需求。

        另外，要注意不要使用任何格式或标记，保持回复自然。可能用户想探索某个主题，或者需要帮助解决某个问题，但目前的信息不足以确定具体需求。因此，最好的做法是礼貌地请求用户进一步澄清他们的需求，以便我能更好地协助他们。
        </think>

        您好！看起来您可能想探索一些有趣的内容呢～能告诉我您想了解什么吗？比如学科知识、趣味问答或者其他类型的内容？我会尽力为您提供帮助！😊  
        """
    completion2 = """<think>
        好的，我现在需要处理用户的查询"D I s c o v E r."。首先，用户可能是在输入某个单词或者短语，但看起来有些字母大写和小写交替，可能是拼写错误或者故意这样写的。接下来，我要分析用户的需求。

        用户可能想查找与"Discover"相关的学科内容，比如词汇解释、例句、相关练习题或者其他学习资源。由于用户输入中没有明确说明具体需求，比如年级、科目或知识点，我需要进一步确认。不过根据之前的规则，如果用户没有提供足够的 参数，应该先调用工具来获取相关信息。

        考虑到用户可能希望得到关于"discover"的学习资料，我应该使用字词查询工具（query_en_word）来获取详细的解释、发音、例句等内容。同时，检查是否有其他工具更适合，比如练习题工具，但用户并没有提到题目，所以暂时排除。

        接下来，我需要构造正确的工具调用参数。根据工具指南，query_en_word的参数包括word，这里应该是"discover"。其他参数如grade、semester等如果没有提供的话就不填，因为它们不是必填项。因此，调用函数时只需传入word:"discover"即可。

        调用工具后，预期返回的内容可能包括单词的音标、词性、释义、例句以及可能的图像或动画资源。根据工具返回的数据，我需要将其整理成适合的响应格式，比如使用<ssw-card>结构，包含print模块和character_svg（如果有的话）。需 要注意的是，用户输入中的大小写可能不影响工具的参数处理，所以保持原样传递即可。

        最后，确保回复符合格式要求，不包含任何禁止的内容，如链接或ID，并且用礼貌专业的语气呈现信息。如果工具返回的内容不足，可能需要提示用户提供更多信息，但目前假设工具能正确返回所需数据。
        </think>

        <ssw-card>
            <print>您输入的"D I s c o v E r."可能存在拼写差异，我为您查询英式英语单词"discover"的相关学习资料：</print>
            <character_svg type="words" content="discover" svg_url="https://static.k12kaoyan.com/learn/english/dictionary/discover/1.svg"/>
            <print>【发音】/dɪˈskʌvə(r)/\n【词性】动词\n【释义】发现；揭示；揭露\n【例句】She discovered a new species of bird in the Amazon rainforest.\n【拓展】近义词：uncover/reveal/spot</print>
        </ssw-card>
        """   
    completion3 = """<think>
        好的，我现在需要处理用户的请求。用户明确提到需要六年级（五四制）的题目，教材版本是人教版（五·四学制），并且主题要贴合教学重点。首先，我得确认用户的需求是否明确，以及是否需要调用工具来获取相关信息。

        首先，用户提到的"五四制"是指中国的一种学制，与传统的"三三二一"学制不同，可能涉及到课程安排和教材版本的不同。用户特别指定了教材版本为人教版，所以需要确保工具调用时使用正确的参数来匹配这个版本。

        接下来，用户需要的是六年级的题目，但五四学制的六年级可能对应的是传统的五年制中的六年级，或者可能有不同的年级划分。需要确认工具是否支持按学制和教材版本来筛选题目。例如，工具可能有参数如grade（年级）、textbook_version（教材版本）、education_system（学制）等。

        然后，用户提到主题要贴合教学重点，这意味着需要获取的题目应该覆盖当前学期的重点知识点。可能需要调用题目搜索工具，参数包括年级、教材版本、学制以及可能的学科和知识点。例如，使用search_tbl_question工具，参数可能包括grade、textbook_version、education_system，以及可能的subject和knowledge_point。

        需要检查工具的参数是否包含这些字段，并且是否为必填项。例如，如果工具的required字段中包含grade和textbook_version，那么必须提供这些参数。而education_system可能不是必填，但用户明确指定了五四制，所以需要包含进去。

        另外，用户可能没有明确提到学科，但根据上下文，可能默认是数学或语文，但为了保险起见，可能需要进一步确认。不过根据用户之前的例子，可能默认是数学，或者需要根据工具的参数来判断。如果工具允许不指定学科，可能需要默认处理，或者在调用工具时留空，让工具自行处理。

        现在，我需要调用search_tbl_question工具，参数包括count（题目数量）、grade（六年级）、textbook_version（人教版）、education_system（五四制），以及可能的subject和knowledge_point。但用户没有明确提到学科和知识点，所以可能需要默认处理，或者工具可能自动根据年级和教材版本来获取相关题目。

        但根据工具的参数定义，可能需要指定subject，比如数学、语文等。用户没有明确说明，但可能需要根据常见的教学重点来推断。例如，六年级数学可能涉及分数、比例、几何等，语文可能涉及阅读理解、作文等。但用户没有具体说明，所以可能需要调用工具时留空，或者根据工具的默认行为处理。

        不过，根据工具的参数要求，可能必须提供subject，所以可能需要进一步询问用户，但根据规则，只有required字段中的参数才需要询问。如果subject不是必填项，那么可以留空，或者根据工具的默认值处理。

        综上所述，正确的工具调用应该是search_tbl_question，参数包括count（用户可能需要一定数量的题目，但用户没有明确说明数量，可能需要默认值，比如5道），grade为六年级，textbook_version为人教版，education_system为五四制。如果这些参数都是必填的，那么需要确保它们被正确填入。

        检查工具的参数定义，假设search_tbl_question的required字段包括grade、textbook_version，而education_system可能不是必填，但用户明确指定了，所以需要包含。因此，调用工具时应包含这些参数，并生成相应的function call。
        </think>

        {"function": {"name": "search_tbl_question", "arguments": {"count": 5, "grade": "六年级", "textbook_version": "人教版", "education_system": "五四学制"}}, "index": 0, "id": "call_1234567890", "type": "function"}
        """
    completion4 = """<think>
        好的，我现在需要处理用户的请求。用户明确提到需要六年级（五四制）的题目，教材版本是人教版（五·四学制），并且主题要贴合教学重点。首先，我得确认用户的需求是否明确，以及是否需要调用工具来获取相关信息。

        首先，用户提到的"五四制"是指中国的一种学制，与传统的"三三二一"学制不同，可能涉及到课程安排和教材版本的不同。用户特别指定了教材版本为人教版，所以需要确保工具调用时使用正确的参数来匹配这个版本。

        接下来，用户需要的是六年级的题目，但五四学制的六年级可能对应的是传统的五年制中的六年级，或者可能有不同的年级划分。需要确认工具是否支持按学制和教材版本来筛选题目。例如，工具可能有参数如grade（年级）、textbook_version（教材版本）、education_system（学制）等。

        然后，用户提到主题要贴合教学重点，这意味着需要获取的题目应该覆盖当前学期的重点知识点。可能需要调用题目搜索工具，参数包括年级、教材版本、学制以及可能的学科和知识点。例如，使用search_tbl_question工具，参数可能包括grade、textbook_version、education_system，以及可能的subject和knowledge_point。

        需要检查工具的参数是否包含这些字段，并且是否为必填项。例如，如果工具的required字段中包含grade和textbook_version，那么必须提供这些参数。而education_system可能不是必填，但用户明确指定了五四制，所以需要包含进去。

        另外，用户可能没有明确提到学科，但根据上下文，可能默认是数学或语文，但为了保险起见，可能需要进一步确认。不过根据用户之前的例子，可能默认是数学，或者需要根据工具的参数来判断。如果工具允许不指定学科，可能需要默认处理，或者在调用工具时留空，让工具自行处理。

        现在，我需要调用search_tbl_question工具，参数包括count（题目数量）、grade（六年级）、textbook_version（人教版）、education_system（五四制），以及可能的subject和knowledge_point。但用户没有明确提到学科和知识点，所以可能需要默认处理，或者工具可能自动根据年级和教材版本来获取相关题目。

        但根据工具的参数定义，可能需要指定subject，比如数学、语文等。用户没有明确说明，但可能需要根据常见的教学重点来推断。例如，六年级数学可能涉及分数、比例、几何等，语文可能涉及阅读理解、作文等。但用户没有具体说明，所以可能需要调用工具时留空，或者根据工具的默认行为处理。

        不过，根据工具的参数要求，可能必须提供subject，所以可能需要进一步询问用户，但根据规则，只有required字段中的参数才需要询问。如果subject不是必填项，那么可以留空，或者根据工具的默认值处理。

        综上所述，正确的工具调用应该是search_tbl_question，参数包括count（用户可能需要一定数量的题目，但用户没有明确说明数量，可能需要默认值，比如5道），grade为六年级，textbook_version为人教版，education_system为五四制。如果这些参数都是必填的，那么需要确保它们被正确填入。

        检查工具的参数定义，假设search_tbl_question的required字段包括grade、textbook_version，而education_system可能不是必填，但用户明确指定了，所以需要包含。因此，调用工具时应包含这些参数，并生成相应的function call。
        </think>
        {"function": {"name": "query_knowledge_card", "arguments": {"knowledge_path": "壶口瀑布", "grade": "八年级下册"}}, "index": 0, "id": "call_9x7vzqk8f3e4rtyg9n8m5hjkl", "type": "function"}
        {"function": {"name": "search_tbl_question", "arguments": {"count": "5", "grade": "六年级", "version": "人教版", "subject": "数学"}}, "index": 0, "id": "call_1234567890", "type": "function"}
        """
    
    # ============== 添加错误测试用例 ==============
    print("=== 测试可能导致 KeyError 的情况 ===")
    
    # 测试用例1: JSON正确但缺少function键
    error_completion1 = """<think>
        测试用例
        </think>

        {"name": "search_tbl_question", "arguments": {"count": 5}}
        """
    
    # 测试用例2: JSON正确但function值不是字典  
    error_completion2 = """<think>
        测试用例
        </think>

        {"function": "not_a_dict", "arguments": {"count": 5}}
        """
    
    # 测试用例3: JSON正确但function字典缺少name键
    error_completion3 = """<think>
        测试用例
        </think>

        {"function": {"arguments": {"count": 5}}, "index": 0}
        """
    
    # 测试用例4: JSON格式错误  
    error_completion4 = """<think>
        测试用例
        </think>

        {"function": {"name": "search_tbl_question", "arguments": {"count": 5}
        """
    
    # 测试用例5: 包含function关键词但不是工具调用
    error_completion5 = """<think>
        I need to use function to solve this problem
        </think>

        根据你的需求，我建议使用function来处理这个问题。function的name应该是xxx，arguments包含各种参数。
        """
    
    error_completions = [error_completion1, error_completion2, error_completion3, error_completion4, error_completion5]
    
    print("测试错误用例...")
    tc_orm = ToolCallAccReward()
    
    for i, completion in enumerate(error_completions, 1):
        try:
            score, status = tc_orm._evaluate_single_completion(completion)
            print(f"错误用例{i}: score={score}, status={status}")
        except Exception as e:
            print(f"错误用例{i}: 抛出异常 {type(e).__name__}: {e}")
    
    print("\n=== 正常测试 ===")

    # completions = [completion1] * 2 + [completion2] * 2 + [completion3] * 3 + [completion4] * 3
    import json
    f=open('/mnt/cfs/ssw/ljc/ms-swift/output_ppo/v65-20250821-195647/completions.jsonl', encoding='utf-8')
    data=[json.loads(line) for line in f]
    completions = [data[1]['completion'][i] for i in range(len(data[1]['completion']))]

    # tc_orm = ToolCallAccReward()
    # score = tc_orm(completions)
    # print(score)

    full_orm = ToolCallsCosineReward(accuracy_orm=ToolCallAccReward)
    score = full_orm(completions)
    print(score)