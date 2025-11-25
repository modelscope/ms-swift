import json
import random
import os


def minimal_json_sampler(input_path: str, output_path: str, count: int):
    """
    从 JSON Lines 文件中随机抽取指定数量的记录并保存。
    假设输入文件是 JSON Lines 格式（每行一个JSON对象）。
    """

    if not os.path.exists(input_path):
        print(f"错误：文件未找到 -> {input_path}")
        return

    # 1. 读取所有记录 (核心)
    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用列表推导式高效读取所有 JSON Lines
        all_records = [json.loads(line.strip()) for line in f if line.strip()]

    # 2. 抽样 (核心)
    # 确保采样数不超过总记录数
    sample_count = min(count, len(all_records))
    sampled_records = random.sample(all_records, sample_count)

    # 3. 保存 (核心)
    with open(output_path, 'w', encoding='utf-8') as f:
        # 保存为格式化的 JSON 数组
        json.dump(sampled_records, f, ensure_ascii=False, indent=4)

    print(f"成功采样 {sample_count} 条记录到 {output_path}")


# --- 运行配置：请修改这三个参数 ---
if __name__ == '__main__':
    # 🎯 您的输入文件路径
    INPUT_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M.json'
    # 🎯 您的输出文件路径
    OUTPUT_DEMO_FILE = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-demo.json'
    # 🎯 您需要的采样条数
    SAMPLE_COUNT = 20

    # --- 请确保您的 INPUT_FILE 存在且是 JSON Lines 格式 ---
    # 执行
    minimal_json_sampler(INPUT_FILE, OUTPUT_DEMO_FILE, SAMPLE_COUNT)