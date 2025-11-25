"""
简单直接的 JSONL null 值清理脚本
"""

import json
import sys
from tqdm import tqdm

# ========== 配置 ==========
INPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-0001.jsonl'
OUTPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-0001-fixed.jsonl'
# ==========================


def check_null(obj):
    """检查对象是否包含 null 值"""
    if obj is None:
        return True

    if isinstance(obj, dict):
        for v in obj.values():
            if check_null(v):
                return True

    elif isinstance(obj, list):
        if len(obj) == 0:
            return True
        for item in obj:
            if check_null(item):
                return True

    elif isinstance(obj, str):
        if len(obj.strip()) == 0:
            return True

    return False


print(f"输入: {INPUT_FILE}")
print(f"输出: {OUTPUT_FILE}\n")

valid = 0
invalid = 0

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        # 先数总行数
        total_lines = sum(1 for _ in f_in)
        print(f"总行数: {total_lines:,}\n")
        f_in.seek(0)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            for line in tqdm(f_in, total=total_lines, desc="处理"):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # 检查是否有 null
                    if check_null(data):
                        invalid += 1
                        continue

                    # 检查必需字段
                    if 'id' not in data or 'image' not in data or 'conversations' not in data:
                        invalid += 1
                        continue

                    # 写入
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    valid += 1

                except Exception as e:
                    invalid += 1
                    if invalid <= 5:
                        print(f"\n解析错误: {e}")

except FileNotFoundError:
    print(f"错误: 文件不存在 {INPUT_FILE}")
    sys.exit(1)

print("\n" + "=" * 50)
print(f"有效: {valid:,}")
print(f"无效: {invalid:,}")
print(f"通过率: {valid/(valid+invalid)*100:.1f}%")
print("=" * 50)
