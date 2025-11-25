# 1. 忍受一次缓慢的加载 (最好已经是 jsonl)
from datasets import load_dataset


input_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.jsonl'
output_file = '/home/dataset/MAmmoTH-VL-Instruct-12M/mammoth_ov_2M-cl.parquet'

print("正在加载 JSONL 文件...")
dataset = load_dataset('json', data_files=input_file)

# 2. 将其保存为 Parquet (或 Arrow 格式)
print("正在保存为 Parquet 格式...")
# dataset.save_to_disk(output_file) # 这是 Arrow 格式，也很快
dataset['train'].to_parquet(output_file) # 直接存为 Parquet 文件