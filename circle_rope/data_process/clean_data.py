import pandas as pd
from datasets import load_dataset
import os

from pyexpat import features

# --- 2. 定义清洗函数 ---

def is_valid_sample(example):
    """
    检查单个样本是否合规。
    如果合规，返回 True；否则返回 False。
    """

    # 规则 1 & 2: 检查 'id' 和 'image' 键是否存在且值不为空
    if not example.get('id') or not example.get('image'):
        # .get('id') 如果键不存在或值为 None，会返回 None (布尔值为 False)
        return False

    # 规则 3: 检查 'conversations' 键是否存在
    if 'conversations' not in example or example['conversations'] is None:
        return False

    # 规则 4: 检查 'conversations' 是否为非空列表
    conversations = example['conversations']
    if not isinstance(conversations, list) or len(conversations) == 0:
        return False

    # 规则 5 & 6: 检查 'conversations' 内部的每个元素
    for turn in conversations:
        # 必须是字典
        if not isinstance(turn, dict):
            return False

        # 必须包含 'from' 和 'value'，且它们的值不能为空
        if not turn.get('from') or not turn.get('value'):
            return False


    if 'video' in example:
        return False

    # 所有检查通过
    return True

def convert_id_to_string(example):
    """
    将 'id' 字段的值强制转换为字符串。
    """
    example['id'] = str(example['id'])
    return example


# --- 3. 加载并执行清理 ---


data_file_path = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json'
cleaned_file_path = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_si_10M-cl.json'

print("\n--- 开始加载数据集 ---")

# 使用 'json' 加载器，指定数据文件
dataset = load_dataset('json', data_files=data_file_path, split='train')

print(f"原始数据集大小: {len(dataset)} 条")
print("原始数据 (Pandas 预览):")
print(dataset.to_pandas())

# 关键步骤：使用 filter() 应用清洗函数
# num_proc=4 可以使用多进程加速，根据您的 CPU 核心数调整
print("\n--- 正在清洗数据... ---")
cleaned_dataset = dataset.filter(is_valid_sample, num_proc=128)

# --- 4. 查看结果 ---
print("\n--- 清洗完成 ---")
print(f"原始数据集大小: {len(dataset)} 条")
print(f"清洗后数据集大小: {len(cleaned_dataset)} 条")
print(f"共“洗掉”了: {len(dataset) - len(cleaned_dataset)} 条不合规数据")

# (新步骤)
# 步骤 2: 转换 (Map)
print("\n--- 正在转换 ID 为字符串 (Map)... ---")
final_dataset = cleaned_dataset.map(
    convert_id_to_string,
    num_proc=128
)

print("\n清洗后的合规数据 (Pandas 预览):")
print(final_dataset.to_pandas())

# --- 5. (可选) 保存清洗后的数据 ---

final_dataset.to_json(cleaned_file_path, orient="records", force_ascii=False, indent=2)
print(f"\n--- 清洗后的数据已保存到 '{cleaned_file_path}' ---")
