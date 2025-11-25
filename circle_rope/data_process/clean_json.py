import json
import os
import pandas as pd  # 注意：Pandas 仅用于创建模拟数据，核心逻辑不依赖它
from tqdm import tqdm


# --- 2. 定义清洗函数 (与上个脚本完全相同) ---

def is_valid_sample(example):
    """
    检查单个样本 (字典) 是否合规。
    如果合规，返回 True；否则返回 False。
    严格过滤所有空值、None、空字符串等无效数据。
    """
    # 规则 1: 检查 'id' (允许 id 为 0，但不能是 None 或空字符串)
    if example.get('id') is None:
        return False
    # 如果 id 是字符串，检查是否为空
    if isinstance(example['id'], str) and not example['id'].strip():
        return False

    # 规则 2: 检查 'image' 字段
    image = example.get('image')
    if image is None:
        return False

    # 如果 image 是字符串，检查是否为空
    if isinstance(image, str):
        if not image.strip():
            return False
    # 如果 image 是列表，检查列表是否为空，以及列表中的每个元素
    elif isinstance(image, list):
        if len(image) == 0:
            return False
        for img in image:
            if img is None or (isinstance(img, str) and not img.strip()):
                return False
    else:
        # image 既不是字符串也不是列表，认为无效
        return False

    # 规则 3: 检查 'conversations' 键是否存在且不为 None
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

        # 检查 'from' 字段
        from_field = turn.get('from')
        if from_field is None:
            return False
        if isinstance(from_field, str) and not from_field.strip():
            return False

        # 检查 'value' 字段
        value_field = turn.get('value')
        if value_field is None:
            return False
        if isinstance(value_field, str) and not value_field.strip():
            return False

    # 规则 7: 不允许有 'video' 字段（只处理图像数据）
    if 'video' in example:
        return False

    return True


# --- 3. 核心逻辑：加载、清洗、转换、保存 (使用 json 库) ---

print("\n--- 开始使用标准 json 库处理 ---")
input_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M.json'
output_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.jsonl'

cleaned_and_transformed_data = []  # 准备一个新列表

# 步骤 1: 加载
with open(input_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

print(f"原始数据集大小: {len(original_data)} 条")

# 步骤 2: 循环、清洗 (Filter) 和 转换 (Map)
for sample in tqdm(original_data):
    # 步骤 2a: 清洗 (Filter)
    if is_valid_sample(sample):
        # 步骤 2b: 转换 (Map) - 仅对合规数据执行
        # 转换 id 为字符串
        sample['id'] = str(sample['id'])

        # 转换 image 为字符串列表
        if isinstance(sample['image'], str):
            sample['image'] = [sample['image']]
        else:
            # 确保列表中所有元素都是字符串
            sample['image'] = [str(img) for img in sample['image']]

        # 转换 conversations 中的所有字段为字符串
        for turn in sample['conversations']:
            turn['from'] = str(turn['from'])
            turn['value'] = str(turn['value'])

        # 删除不需要的字段
        if 'old_id' in sample:
            sample.pop('old_id')

        # 二次验证：确保没有 None 值（双重保险）
        try:
            json_str = json.dumps(sample, ensure_ascii=False)
            if 'null' in json_str.lower():
                continue  # 跳过包含 null 的样本
        except:
            continue  # 跳过无法序列化的样本

        # 步骤 2c: 添加到新列表
        cleaned_and_transformed_data.append(sample)

print(f"清洗后数据集大小: {len(cleaned_and_transformed_data)} 条")
print(f"共“洗掉”了: {len(original_data) - len(cleaned_and_transformed_data)} 条不合规数据")


# --- 5. 保存结果 ---
print(f"\n--- 正在保存最终数据到 '{output_file}' ---")
with open(output_file, 'w', encoding='utf-8') as f:
    # 保存为 JSONL 格式（每行一个 JSON 对象）
    # ensure_ascii=False 确保中文等字符正确写入
    for sample in cleaned_and_transformed_data:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')



print("--- 处理完成 ---")
