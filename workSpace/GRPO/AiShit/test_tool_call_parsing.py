#!/usr/bin/env python3
"""
测试脚本：验证GRPO训练中的工具调用解析是否正常工作
"""

import sys
import os
sys.path.insert(0, '/mnt/cfs/ssw/ljc/ms-swift')

from swift.plugin.agent_template.qwen import QwenEnAgentTemplate, QwenZhAgentTemplate
from swift.plugin.agent_template.react import ReactEnAgentTemplate
from swift.plugin.agent_template.hermes import HermesAgentTemplate
from swift.llm.infer import Function


def test_agent_templates():
    """测试不同Agent Template的工具调用解析"""
    
    print("🔍 测试Agent Template工具调用解析")
    print("=" * 60)
    
    # 测试数据
    test_cases = [
        {
            "name": "Qwen英文格式",
            "template": QwenEnAgentTemplate(),
            "response": """I need to search for information.

✿FUNCTION✿: chinese_dictation
✿ARGS✿: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}
✿RESULT✿: """
        },
        {
            "name": "Qwen中文格式", 
            "template": QwenZhAgentTemplate(),
            "response": """我需要搜索相关信息。

✿FUNCTION✿: chinese_dictation
✿ARGS✿: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}
✿RESULT✿: """
        },
        {
            "name": "React英文格式",
            "template": ReactEnAgentTemplate(), 
            "response": """Thought: I need to search for information.
Action: chinese_dictation
Action Input: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}
Observation: """
        },
        {
            "name": "Hermes格式",
            "template": HermesAgentTemplate(),
            "response": """I'll help you search for information.

<tool_call>
{"name": "chinese_dictation", "arguments": {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}}
</tool_call>"""
        }
    ]
    
    # 执行测试
    for test_case in test_cases:
        print(f"\n📝 测试: {test_case['name']}")
        print("-" * 40)
        
        try:
            functions = test_case['template'].get_toolcall(test_case['response'])
            
            if functions:
                print(f"✅ 成功解析到 {len(functions)} 个工具调用:")
                for i, func in enumerate(functions):
                    print(f"   {i+1}. 工具名: {func.name}")
                    print(f"      参数: {func.arguments}")
            else:
                print("❌ 未能解析到工具调用")
                
        except Exception as e:
            print(f"❌ 解析失败: {str(e)}")


def test_plugin_reward():
    """测试修复后的插件奖励函数"""
    
    print("\n\n🎯 测试插件奖励函数")
    print("=" * 60)
    
    # 导入修复后的插件
    try:
        from plugin_fixed import ToolCallAccReward
        reward_func = ToolCallAccReward()
        
        # 测试不同格式的completion
        test_completions = [
            {
                "name": "Qwen格式",
                "completion": """我需要搜索相关信息。

✿FUNCTION✿: chinese_dictation
✿ARGS✿: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}
✿RESULT✿: """
            },
            {
                "name": "React格式", 
                "completion": """Thought: I need to search for information.
Action: chinese_dictation
Action Input: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}
Observation: """
            },
            {
                "name": "JSON格式",
                "completion": """<think>我需要调用工具来搜索</think>

{"function": {"name": "chinese_dictation", "arguments": {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35"}}}"""
            },
            {
                "name": "无工具调用",
                "completion": "这是一个普通的回答，没有包含任何工具调用。"
            }
        ]
        
        # 测试每个completion
        for test_case in test_completions:
            print(f"\n📝 测试: {test_case['name']}")
            print("-" * 40)
            
            try:
                rewards = reward_func([test_case['completion']])
                reward = rewards[0] if rewards else 0.0
                
                if reward > 0.8:
                    print(f"✅ 高奖励: {reward:.2f} (工具调用成功)")
                elif reward > 0.2:
                    print(f"⚠️  中等奖励: {reward:.2f} (工具调用有问题)")
                else:
                    print(f"❌ 低奖励: {reward:.2f} (无工具调用或失败)")
                    
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                
    except ImportError as e:
        print(f"❌ 无法导入修复后的插件: {str(e)}")
        print("请确保 plugin_fixed.py 文件在当前目录中")


def test_vllm_output_parsing():
    """测试vLLM输出的解析"""
    
    print("\n\n🔧 测试vLLM输出解析")
    print("=" * 60)
    
    # 模拟vLLM的实际输出
    mock_vllm_response = """<think>
好的，我现在需要处理用户的请求。用户提到他们即将参加八年级下册的期中考试，担心听写部分可能不及格，所以请求帮忙寻找35个来自《壶口瀑布》这篇课文的核心词汇。
</think>

✿FUNCTION✿: chinese_dictation
✿ARGS✿: {"grade": "八年级", "term": "下册", "text_title": "壶口瀑布", "count": "35", "book_version": "统编版"}
✿RESULT✿: """
    
    print("模拟的vLLM输出:")
    print(mock_vllm_response)
    print("\n解析结果:")
    
    # 使用Qwen模板解析
    qwen_template = QwenEnAgentTemplate()
    try:
        functions = qwen_template.get_toolcall(mock_vllm_response)
        if functions:
            print(f"✅ 成功解析到工具调用:")
            for func in functions:
                print(f"   工具名: {func.name}")
                print(f"   参数: {func.arguments}")
        else:
            print("❌ 未能解析到工具调用")
    except Exception as e:
        print(f"❌ 解析失败: {str(e)}")


def main():
    """主函数"""
    print("🚀 GRPO工具调用测试开始")
    print("测试目标：验证Agent Template和插件奖励函数是否正常工作\n")
    
    # 运行所有测试
    test_agent_templates()
    test_plugin_reward() 
    test_vllm_output_parsing()
    
    print("\n\n📋 总结和建议:")
    print("=" * 60)
    print("1. ✅ 使用修复后的训练脚本: GRPO_4B_tool_call_fixed.sh")
    print("2. ✅ 添加 --agent_template qwen_en 参数")
    print("3. ✅ 使用修复后的插件: plugin_fixed.py") 
    print("4. ✅ 确保数据集格式正确: 参考 dataset_format_example.json")
    print("5. ⚠️  监控训练日志中的工具调用状态")
    print("\n🎯 如果测试通过，您的GRPO训练应该能正确处理工具调用了！")


if __name__ == "__main__":
    main() 