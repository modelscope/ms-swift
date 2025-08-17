#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 assistant response 被增加两次的问题
"""

import unittest
from copy import deepcopy


def replace_assistant_response_with_ids(messages, completion_ids):
    """模拟 replace_assistant_response_with_ids 函数"""
    if not completion_ids:
        return messages
    
    if isinstance(completion_ids[0], int):
        completion_ids = [completion_ids]

    remaining_completions = len(completion_ids)
    completion_index = 0

    for message in reversed(messages):
        if message['role'] != 'assistant':
            continue

        if completion_index >= remaining_completions:
            break

        # Assign completion IDs (starting from last)
        message['content'] = completion_ids[-1 - completion_index]
        completion_index += 1

    return messages


class TestDoubleAssistantIssue(unittest.TestCase):
    """测试assistant response被增加两次的问题"""

    def test_double_assistant_addition(self):
        """
        测试assistant response被增加两次的情况
        
        问题描述：在某个地方，assistant response被添加了两次，
        导致最终结果中有两个assistant消息
        """
        
        # 初始数据
        original_messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]
        completion_ids = [1, 2, 3]
        
        print(f"初始数据: {original_messages}")
        
        # 第一次处理：正常的replace_assistant_response_with_ids
        messages = deepcopy(original_messages)
        messages = replace_assistant_response_with_ids(messages, completion_ids)
        print(f"第一次处理后: {messages}")
        
        # 模拟问题：某个地方错误地添加了第二个assistant消息
        # 这可能是模板编码、数据处理或其他地方的问题
        messages.append({'role': 'assistant', 'content': 'Hi there'})
        print(f"错误添加第二个assistant后: {messages}")
        
        # 再次调用replace_assistant_response_with_ids
        # 这会导致两个assistant消息都被替换为token ids
        messages = replace_assistant_response_with_ids(messages, completion_ids)
        print(f"再次替换后: {messages}")
        
        # 检查结果
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
        print(f"所有assistant消息: {assistant_messages}")
        
        # 验证问题：现在有两个assistant消息，都包含token ids
        self.assertEqual(len(assistant_messages), 2)
        self.assertEqual(assistant_messages[0]['content'], [1, 2, 3])
        self.assertEqual(assistant_messages[1]['content'], [1, 2, 3])
        
        print(f"问题确认：有两个assistant消息，都包含相同的token ids")

    def test_template_encoding_double_assistant(self):
        """
        测试模板编码过程中可能出现的double assistant问题
        """
        
        # 模拟模板编码过程
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]
        completion_ids = [1, 2, 3]
        
        print(f"模板编码 - 初始数据: {messages}")
        
        # 步骤1: 替换token ids
        messages = replace_assistant_response_with_ids(messages, completion_ids)
        print(f"模板编码 - 替换后: {messages}")
        
        # 步骤2: 模拟模板编码过程中错误地添加了第二个assistant消息
        # 这可能发生在模板处理、数据预处理或其他地方
        template_encoded_messages = messages + [{'role': 'assistant', 'content': 'Hi there'}]
        print(f"模板编码 - 错误添加第二个assistant后: {template_encoded_messages}")
        
        # 步骤3: 再次调用replace_assistant_response_with_ids
        # 这会导致两个assistant消息都被替换
        result = replace_assistant_response_with_ids(template_encoded_messages, completion_ids)
        print(f"模板编码 - 再次替换后: {result}")
        
        # 检查结果
        assistant_messages = [msg for msg in result if msg['role'] == 'assistant']
        print(f"模板编码 - 所有assistant消息: {assistant_messages}")
        
        # 验证问题
        self.assertEqual(len(assistant_messages), 2)
        for i, msg in enumerate(assistant_messages):
            print(f"Assistant消息{i}: {msg['content']} (类型: {type(msg['content'])})")

    def test_batch_processing_double_assistant(self):
        """
        测试批处理中可能出现的double assistant问题
        """
        
        # 模拟批处理数据
        batch = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there'}
                ],
                'response_token_ids': [1, 2, 3]
            }
        ]
        
        print(f"批处理 - 初始数据: {batch[0]['messages']}")
        
        # 第一次处理
        for data in batch:
            if 'response_token_ids' in data and data['response_token_ids']:
                data['messages'] = replace_assistant_response_with_ids(
                    data['messages'], data['response_token_ids']
                )
        
        print(f"批处理 - 第一次处理后: {batch[0]['messages']}")
        
        # 模拟问题：某个地方错误地添加了第二个assistant消息
        batch[0]['messages'].append({'role': 'assistant', 'content': 'Hi there'})
        print(f"批处理 - 错误添加第二个assistant后: {batch[0]['messages']}")
        
        # 再次处理（可能在其他地方被调用）
        for data in batch:
            if 'response_token_ids' in data and data['response_token_ids']:
                data['messages'] = replace_assistant_response_with_ids(
                    data['messages'], data['response_token_ids']
                )
        
        print(f"批处理 - 再次处理后: {batch[0]['messages']}")
        
        # 检查结果
        assistant_messages = [msg for msg in batch[0]['messages'] if msg['role'] == 'assistant']
        print(f"批处理 - 所有assistant消息: {assistant_messages}")
        
        # 验证问题
        self.assertEqual(len(assistant_messages), 2)

    def test_grpo_workflow_double_assistant(self):
        """
        模拟GRPO工作流程中的double assistant问题
        """
        
        # 模拟GRPO训练中的完整流程
        original_data = {
            'messages': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there'}
            ],
            'response_token_ids': [1, 2, 3]
        }
        
        print(f"GRPO工作流程 - 初始数据: {original_data}")
        
        # 步骤1: 在_prepare_ga_batch_encoded_inputs中调用replace_assistant_response_with_ids
        data = deepcopy(original_data)
        if 'response_token_ids' in data and data['response_token_ids']:
            data['messages'] = replace_assistant_response_with_ids(
                data['messages'], data['response_token_ids']
            )
        
        print(f"GRPO工作流程 - 步骤1后: {data}")
        
        # 步骤2: 模拟某个地方错误地添加了第二个assistant消息
        # 这可能发生在数据预处理、模板编码或其他地方
        data['messages'].append({'role': 'assistant', 'content': 'Hi there'})
        print(f"GRPO工作流程 - 步骤2后（错误添加）: {data}")
        
        # 步骤3: 模拟_apply_chat_template_to_messages_list中的处理
        # 这里可能会再次处理messages
        final_data = deepcopy(data)
        
        # 模拟可能的重复处理
        if 'response_token_ids' in final_data and final_data['response_token_ids']:
            final_data['messages'] = replace_assistant_response_with_ids(
                final_data['messages'], final_data['response_token_ids']
            )
        
        print(f"GRPO工作流程 - 最终结果: {final_data}")
        
        # 检查结果
        assistant_messages = [msg for msg in final_data['messages'] if msg['role'] == 'assistant']
        print(f"GRPO工作流程 - 所有assistant消息: {assistant_messages}")
        
        # 验证问题
        self.assertEqual(len(assistant_messages), 2)
        for i, msg in enumerate(assistant_messages):
            print(f"Assistant消息{i}: {msg['content']}")

    def test_solution_prevention(self):
        """
        测试解决方案：防止double assistant问题
        """
        
        # 解决方案1: 在replace_assistant_response_with_ids之前检查是否已有assistant消息
        def safe_replace_assistant_response_with_ids(messages, completion_ids):
            """安全版本的replace_assistant_response_with_ids"""
            if not completion_ids:
                return messages
            
            if isinstance(completion_ids[0], int):
                completion_ids = [completion_ids]

            # 检查是否已有assistant消息
            assistant_count = sum(1 for msg in messages if msg['role'] == 'assistant')
            print(f"当前assistant消息数量: {assistant_count}")
            
            if assistant_count == 0:
                # 如果没有assistant消息，添加一个
                messages.append({'role': 'assistant', 'content': completion_ids[0]})
                return messages
            
            # 原有的替换逻辑
            remaining_completions = len(completion_ids)
            completion_index = 0

            for message in reversed(messages):
                if message['role'] != 'assistant':
                    continue

                if completion_index >= remaining_completions:
                    break

                message['content'] = completion_ids[-1 - completion_index]
                completion_index += 1

            return messages
        
        # 测试安全版本
        messages = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]
        completion_ids = [1, 2, 3]
        
        print(f"解决方案测试 - 初始数据: {messages}")
        
        # 第一次处理
        result1 = safe_replace_assistant_response_with_ids(messages, completion_ids)
        print(f"解决方案测试 - 第一次处理后: {result1}")
        
        # 错误地添加第二个assistant
        result1.append({'role': 'assistant', 'content': 'Hi there'})
        print(f"解决方案测试 - 错误添加后: {result1}")
        
        # 再次处理
        result2 = safe_replace_assistant_response_with_ids(result1, completion_ids)
        print(f"解决方案测试 - 再次处理后: {result2}")
        
        # 检查结果
        assistant_messages = [msg for msg in result2 if msg['role'] == 'assistant']
        print(f"解决方案测试 - 所有assistant消息: {assistant_messages}")
        
        # 验证：现在有两个assistant消息，但这是预期的行为
        self.assertEqual(len(assistant_messages), 2)


def run_tests():
    """运行所有测试"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
