import json

# 检查 JSONL 文件前 100 行，查找 null 值
input_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.jsonl'

print("检查前 100 行数据中的 null 值...\n")

with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000000:
            break

        data = json.loads(line)

        # 检查所有字段
        has_null = False
        null_fields = []

        for key, value in data.items():
            if value is None:
                has_null = True
                null_fields.append(key)
            elif isinstance(value, list):
                if None in value:
                    has_null = True
                    null_fields.append(f"{key}[contains None]")
            elif isinstance(value, dict):
                for k, v in value.items():
                    if v is None:
                        has_null = True
                        null_fields.append(f"{key}.{k}")

        if has_null:
            print(f"第 {i+1} 行包含 null: {null_fields}")
            print(f"  数据: {json.dumps(data, ensure_ascii=False)[:200]}...\n")

print("检查完成！")
