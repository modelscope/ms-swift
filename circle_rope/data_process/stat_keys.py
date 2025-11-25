"""
统计 JSONL 文件中所有出现过的键值
"""

import json
from collections import Counter
from tqdm import tqdm

# ========== 配置 ==========
INPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-01.jsonl'
MAX_LINES = None  # None = 全部读取，或设置数字如 10000
# ==========================


def get_all_keys(obj, prefix=''):
    """递归获取对象中所有键"""
    keys = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)

            # 递归获取嵌套的键
            nested_keys = get_all_keys(v, full_key)
            keys.update(nested_keys)

    elif isinstance(obj, list):
        if len(obj) > 0:
            # 检查列表第一个元素（假设列表元素结构一致）
            nested_keys = get_all_keys(obj[0], f"{prefix}[0]")
            keys.update(nested_keys)

    return keys


print(f"读取文件: {INPUT_FILE}\n")

all_keys = set()
key_counts = Counter()
total_lines = 0
example_data = None

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="扫描")):
            if MAX_LINES and i >= MAX_LINES:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 保存第一个样本作为示例
                if example_data is None:
                    example_data = data

                # 获取当前样本的所有键
                keys = get_all_keys(data)
                all_keys.update(keys)

                # 统计每个键出现的次数
                for key in keys:
                    key_counts[key] += 1

                total_lines += 1

            except Exception as e:
                if i < 5:
                    print(f"第 {i+1} 行解析错误: {e}")

except FileNotFoundError:
    print(f"错误: 文件不存在 {INPUT_FILE}")
    exit(1)

# 输出结果
print("\n" + "=" * 60)
print(f"总样本数: {total_lines:,}")
print("=" * 60)

print("\n所有键值（按出现次数排序）:")
print("-" * 60)
for key, count in key_counts.most_common():
    percentage = count / total_lines * 100
    print(f"{key:40} {count:>8,} 条  ({percentage:>6.2f}%)")

print("\n" + "=" * 60)
print(f"唯一键数量: {len(all_keys)}")
print("=" * 60)

# 输出示例数据
if example_data:
    print("\n第一个样本示例:")
    print("-" * 60)
    print(json.dumps(example_data, indent=2, ensure_ascii=False)[:500])
    print("...")
