import glob
import tarfile
import os

# 1. 定义路径
SOURCE_DIR = "/home/dataset/MAmmoTH-VL-Instruct-12M/multi_image_data"
DEST_DIR = "/home/dataset/MAmmoTH-VL-Instruct-12M/si-img"

# 2. 确保目标目录存在 (如果不存在就创建)
print(f"确保目标目录存在: {DEST_DIR}")
os.makedirs(DEST_DIR, exist_ok=True)

# 3. 构造搜索路径并查找所有 tar.gz 文件
search_path = os.path.join(SOURCE_DIR, 'shard_*.tar.gz')
file_list = glob.glob(search_path)
file_list.sort()  # 保证解压顺序

if not file_list:
    print(f"错误：在 {SOURCE_DIR} 中未找到 'shard_*.tar.gz' 文件。")
else:
    print(f"在 {SOURCE_DIR} 找到了 {len(file_list)} 个文件。")
    print("开始解压...")

    # 4. 循环解压
    for f_path in file_list:
        # os.path.basename 只显示文件名，让输出更干净
        print(f"  正在处理: {os.path.basename(f_path)}")

        # 'r:gz' 表示读取 gzip 压缩的 tar 包
        with tarfile.open(f_path, "r:gz") as tar:
            # 将所有内容解压到 DEST_DIR
            tar.extractall(path=DEST_DIR)

    print(f"全部解压完成。所有文件都在 {DEST_DIR} 中。")