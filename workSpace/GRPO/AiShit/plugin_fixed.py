import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolCallAccReward:
    """修复版本的工具调用奖励函数，支持Qwen Agent Template格式"""
    
    def __init__(self):
        self.suc_score = 1.0  # 成功分数
        self.fail_score = 0.3  # 失败分数  
        self.ntc_score = 0.0  # 无工具调用分数
        
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            try:
                reward, status = self._evaluate_single_completion(completion)
                rewards.append(reward)
                logger.info(f"工具调用状态: {status}, 奖励: {reward}")
            except Exception as e:
                logger.error(f"评估过程中出现错误: {e}")
                rewards.append(self.ntc_score)
        return rewards
    
    def _evaluate_single_completion(self, completion: str) -> Tuple[float, str]:
        """
        评估单个completion的奖励分数，支持多种格式
        
        Args:
            completion: 模型生成的回复内容
            
        Returns:
            Tuple[float, str]: (奖励分数, 状态描述)
        """
        
        # 1. 首先尝试解析Qwen Agent Template格式
        qwen_result = self._parse_qwen_format(completion)
        if qwen_result:
            return qwen_result
            
        # 2. 尝试解析React格式  
        react_result = self._parse_react_format(completion)
        if react_result:
            return react_result
            
        # 3. 尝试解析JSON格式（原有逻辑）
        json_result = self._parse_json_format(completion)
        if json_result:
            return json_result
            
        # 4. 如果都没有匹配，检查是否包含思考内容
        if self._has_thinking_content(completion):
            return self.ntc_score, "有思考但无工具调用"
            
        return self.ntc_score, "无工具调用"
    
    def _parse_qwen_format(self, completion: str) -> Optional[Tuple[float, str]]:
        """解析Qwen Agent Template格式：✿FUNCTION✿: 和 ✿ARGS✿:"""
        try:
            # 查找✿FUNCTION✿:模式
            function_pattern = r'✿FUNCTION✿:\s*([^\n]+)'
            args_pattern = r'✿ARGS✿:\s*([^\n✿]+)'
            
            function_match = re.search(function_pattern, completion)
            args_match = re.search(args_pattern, completion)
            
            if function_match and args_match:
                tool_name = function_match.group(1).strip()
                tool_args = args_match.group(1).strip()
                
                # 验证工具参数
                is_valid, error_msg = self._validate_tool_call(tool_name, tool_args)
                if is_valid:
                    return self.suc_score, f"Qwen格式工具调用成功: {tool_name}"
                else:
                    return self.fail_score, f"Qwen格式工具调用失败: {error_msg}"
                    
        except Exception as e:
            logger.debug(f"Qwen格式解析失败: {e}")
            
        return None
    
    def _parse_react_format(self, completion: str) -> Optional[Tuple[float, str]]:
        """解析React格式：Action: 和 Action Input:"""
        try:
            action_pattern = r'Action:\s*([^\n]+)'
            input_pattern = r'Action Input:\s*([^\n]+)'
            
            action_match = re.search(action_pattern, completion)
            input_match = re.search(input_pattern, completion)
            
            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_args = input_match.group(1).strip()
                
                # 验证工具参数
                is_valid, error_msg = self._validate_tool_call(tool_name, tool_args)
                if is_valid:
                    return self.suc_score, f"React格式工具调用成功: {tool_name}"
                else:
                    return self.fail_score, f"React格式工具调用失败: {error_msg}"
                    
        except Exception as e:
            logger.debug(f"React格式解析失败: {e}")
            
        return None
    
    def _parse_json_format(self, completion: str) -> Optional[Tuple[float, str]]:
        """解析JSON格式（原有逻辑）"""
        try:
            # 首先处理<think>标签
            think_pattern = re.compile(r'<think>(.*?)</think>\n\n(.*)', re.DOTALL)
            think_match = think_pattern.search(completion)
            
            if think_match:
                content = think_match.group(2)
            else:
                content = completion
                
            # 检查是否包含JSON工具调用的关键词
            check_words = ['function', 'name', 'arguments']
            if all(word in content for word in check_words):
                tool_content = json.loads(content)
                tool_name = tool_content['function']['name']
                tool_args = tool_content['function']['arguments']
                
                is_valid, error_msg = self._validate_tool_call(tool_name, tool_args)
                if is_valid:
                    return self.suc_score, f"JSON格式工具调用成功: {tool_name}"
                else:
                    return self.fail_score, f"JSON格式工具调用失败: {error_msg}"
                    
        except json.JSONDecodeError:
            logger.debug("JSON格式解析失败")
        except KeyError as e:
            logger.debug(f"JSON格式缺少关键字段: {e}")
        except Exception as e:
            logger.debug(f"JSON格式解析异常: {e}")
            
        return None
    
    def _validate_tool_call(self, tool_name: str, tool_args: str) -> Tuple[bool, str]:
        """验证工具调用的有效性"""
        try:
            # 检查工具名称是否为空
            if not tool_name or tool_name.strip() == "":
                return False, "工具名称为空"
                
            # 尝试解析参数（如果是JSON格式）
            if tool_args.strip().startswith('{'):
                try:
                    args_dict = json.loads(tool_args)
                    if not isinstance(args_dict, dict):
                        return False, "参数不是有效的字典格式"
                except json.JSONDecodeError:
                    return False, "参数不是有效的JSON格式"
            
            # 这里可以添加更多具体的工具验证逻辑
            # 例如检查特定工具的必需参数等
            
            return True, "验证通过"
            
        except Exception as e:
            return False, f"验证过程出错: {str(e)}"
    
    def _has_thinking_content(self, completion: str) -> bool:
        """检查是否包含思考内容"""
        think_indicators = ['<think>', 'Thought:', '我需要', '让我想想', '分析一下']
        return any(indicator in completion for indicator in think_indicators)


# 兼容性：保持原有的类名
class ExternalToolCallCombinedCosineReward(ToolCallAccReward):
    """为了兼容性保留的类名别名"""
    pass


# 注册奖励函数
def external_tool_call_combined_cosine():
    """工厂函数，返回工具调用奖励实例"""
    return ToolCallAccReward()


if __name__ == "__main__":
    # 测试不同格式的工具调用
    reward_func = ToolCallAccReward()
    
    # 测试Qwen格式
    qwen_completion = """
    我需要搜索相关信息。
    
    ✿FUNCTION✿: search_question
    ✿ARGS✿: {"query": "语文题目", "grade": "八年级"}
    ✿RESULT✿: 
    """
    
    # 测试React格式
    react_completion = """
    Thought: I need to search for information.
    Action: search_question
    Action Input: {"query": "语文题目", "grade": "八年级"}
    Observation:
    """
    
    # 测试JSON格式
    json_completion = """
    <think>我需要调用工具</think>
    
    {"function": {"name": "search_question", "arguments": {"query": "语文题目", "grade": "八年级"}}}
    """
    
    completions = [qwen_completion, react_completion, json_completion]
    rewards = reward_func(completions)
    
    print("测试结果:")
    for i, (completion, reward) in enumerate(zip(completions, rewards)):
        print(f"格式 {i+1}: 奖励 = {reward}") 