"""
只保留指定字段的 JSONL 清理脚本
"""

import json
from tqdm import tqdm

# ========== 配置 ==========
INPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-01-fix.jsonl'
OUTPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-01-clean.jsonl'

# 要保留的字段
KEEP_FIELDS = ['conversations', 'image', 'id']
# ==========================


print(f"输入: {INPUT_FILE}")
print(f"输出: {OUTPUT_FILE}")
print(f"保留字段: {KEEP_FIELDS}\n")

total = 0
kept = 0
removed_fields = set()

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

    for line in tqdm(f_in, desc="处理"):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
            total += 1

            # 记录被删除的字段
            for key in data.keys():
                if key not in KEEP_FIELDS:
                    removed_fields.add(key)

            # 只保留指定字段
            cleaned_data = {k: data[k] for k in KEEP_FIELDS if k in data}

            # 检查必需字段是否都存在
            if all(k in cleaned_data for k in KEEP_FIELDS):
                f_out.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')
                kept += 1
            else:
                missing = [k for k in KEEP_FIELDS if k not in cleaned_data]
                if total <= 5:
                    print(f"跳过第 {total} 行: 缺少字段 {missing}")

        except Exception as e:
            if total <= 5:
                print(f"第 {total} 行解析错误: {e}")

print("\n" + "=" * 60)
print(f"处理完成！")
print("=" * 60)
print(f"总样本数: {total:,}")
print(f"保留样本: {kept:,}")
print(f"跳过样本: {total - kept:,}")

if removed_fields:
    print(f"\n删除的字段: {sorted(removed_fields)}")

print("=" * 60)
