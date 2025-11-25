import json
import os

# 测试文件
test_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/merged_data-01.jsonl'

print(f"检查文件: {test_file}")
print(f"文件存在: {os.path.exists(test_file)}")

if os.path.exists(test_file):
    print(f"文件大小: {os.path.getsize(test_file) / 1024 / 1024:.2f} MB")

    print("\n读取前 5 行:")
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            print(f"第 {i+1} 行长度: {len(line)}")
            try:
                data = json.loads(line)
                print(f"  ✓ JSON 有效, keys: {list(data.keys())}")
            except Exception as e:
                print(f"  ✗ JSON 错误: {e}")
else:
    print("\n可能的原因:")
    print("1. 文件路径错误")
    print("2. 文件还没生成")
    print("3. 权限问题")

    # 列出目录内容
    dir_path = '/home/dataset/MAmmoTH-VL-Instruct-12M/'
    if os.path.exists(dir_path):
        print(f"\n目录内容 ({dir_path}):")
        for f in os.listdir(dir_path)[:20]:
            print(f"  - {f}")
