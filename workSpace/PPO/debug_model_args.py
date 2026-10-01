#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/mnt/cfs/ssw/ljc/ms-swift')

from swift.llm.argument import RlhfArguments

# 模拟命令行参数
test_args = [
    '--rlhf_type', 'ppo',
    '--model', '/mnt/cfs/ssw/ljc/LLaMA-Factory/models/Qwen3-4B',
    '--model_type', 'qwen3',
    '--reward_model', 'orm://CombinedCosineReward',
    '--train_type', 'full',
]

print("测试RLHF参数解析...")
try:
    args = RlhfArguments.from_pretrained(test_args)
    print(f"✅ 参数解析成功!")
    print(f"  model: {getattr(args, 'model', 'NOT_SET')}")
    print(f"  reward_model: {getattr(args, 'reward_model', 'NOT_SET')}")
    print(f"  ref_model: {getattr(args, 'ref_model', 'NOT_SET')}")
    print(f"  value_model: {getattr(args, 'value_model', 'NOT_SET')}")
except Exception as e:
    print(f"❌ 参数解析失败: {e}")
    import traceback
    traceback.print_exc() 