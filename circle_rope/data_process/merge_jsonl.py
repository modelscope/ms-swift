"""
合并、打乱、采样 JSONL 文件

用法：
    python merge_jsonl.py

配置：
    修改脚本中的 CONFIG 部分
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

# ==================== 配置区 ====================

CONFIG = {
    # 输入文件列表
    "input_files": [
        "/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.jsonl",
        "/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_si_10M-cl.jsonl",
        # 添加更多文件...
    ],

    # 输出文件路径
    "output_file": "/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-01.jsonl",

    # 采样比例 (0-1)，None 表示使用全部数据
    "sample_ratio": 0.1,  # 例如：保留 10% 的数据

    # 随机种子（保证可复现）
    "random_seed": 42,

    # 是否打乱数据
    "shuffle": True,
}

# ==================== 主逻辑 ====================

def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"加载 {Path(file_path).name}"):
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"保存到 {Path(file_path).name}"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("合并、打乱、采样 JSONL 文件")
    print("=" * 60)

    # 设置随机种子
    if CONFIG["random_seed"] is not None:
        random.seed(CONFIG["random_seed"])
        print(f"\n随机种子: {CONFIG['random_seed']}")

    # 1. 加载所有文件
    print("\n步骤 1: 加载输入文件")
    all_data = []
    file_stats = []

    for file_path in CONFIG["input_files"]:
        print(f"\n加载: {file_path}")
        data = load_jsonl(file_path)
        file_stats.append((Path(file_path).name, len(data)))
        all_data.extend(data)

    print(f"\n文件统计:")
    for filename, count in file_stats:
        print(f"  - {filename}: {count:,} 条")
    print(f"  总计: {len(all_data):,} 条")

    # 2. 打乱数据
    if CONFIG["shuffle"]:
        print(f"\n步骤 2: 打乱数据")
        random.shuffle(all_data)
        print(f"  已打乱 {len(all_data):,} 条数据")

    # 3. 采样
    if CONFIG["sample_ratio"] is not None and CONFIG["sample_ratio"] < 1.0:
        print(f"\n步骤 3: 采样数据")
        sample_size = int(len(all_data) * CONFIG["sample_ratio"])
        sampled_data = all_data[:sample_size]
        print(f"  采样比例: {CONFIG['sample_ratio']:.2%}")
        print(f"  采样数量: {len(sampled_data):,} / {len(all_data):,} 条")
    else:
        sampled_data = all_data
        print(f"\n步骤 3: 使用全部数据 ({len(sampled_data):,} 条)")

    # 4. 保存结果
    print(f"\n步骤 4: 保存结果")
    save_jsonl(sampled_data, CONFIG["output_file"])

    print("\n" + "=" * 60)
    print(f"完成！输出文件: {CONFIG['output_file']}")
    print(f"最终数据量: {len(sampled_data):,} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
