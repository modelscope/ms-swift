# convert_libero_to_vla0_format.py
import json
import os
import pathlib
import shutil

import numpy as np
import tensorflow_datasets as tfds
import tyro
from PIL import Image
from tqdm import tqdm

# 定义将要合并处理的 LIBERO 数据集子集
RAW_DATASET_NAMES = [
    # "libero_10_no_noops",
    # "libero_goal_no_noops",
    # "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# --- VLA-0 格式化模板 ---
# H, D, B 将在脚本中动态填充
# H: 预测的时间步长 (Action chunking size), 这里简化为1
# D: 动作维度
# B: 量化后的整数范围上限
SYSTEM_PROMPT_TEMPLATE = (
    "Analyze the input image and predict robot actions for the next {H} timesteps. "
    "Each action has {D} dimensions. Output a single sequence of {H_times_D} integers "
    "(0 - {B} each), representing the {H} timesteps sequentially. "
    "Provide only space-separated numbers. Nothing else."
)

# 包含两个图像占位符
PROMPT_TEMPLATE = "task description: {instruction}\n\nimage:\n<image>\n<image>"


def main(
    data_dir: str = "/home/yuquan002/ssd/modified_libero_rlds",
    output_dir: str = "/home/yuquan002/ssd/libero_vl_dataset",
    action_quantization_bins: int = 1000,
):
    """
    将 LIBERO RLDS 数据集转换为适用于 Qwen3-VL-4B 微调的 JSONL 格式。

    Args:
        data_dir: 包含原始 LIBERO RLDS 数据集的目录。
        output_dir: 输出转换后数据的目录。
        action_quantization_bins: 动作量化的整数范围上限。
    """
    output_path = pathlib.Path(output_dir) / RAW_DATASET_NAMES[0].replace("no_noops", "vla0")
    images_path = output_path / "images"

    # 清理并创建输出目录
    print(f"将数据输出到: {output_path}")
    if output_path.exists():
        print("警告: 输出目录已存在，将进行覆盖。")
        shutil.rmtree(output_path)
    images_path.mkdir(parents=True, exist_ok=True)
    
    # 填充系统提示模板
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        H=1,  # 预测 1 个时间步长
        D=7,
        H_times_D=7,  # H=1, D=7
        B=action_quantization_bins
    )
    print("系统提示示例:")
    print(system_prompt)
    print("-" * 30)

    '''
    Action normalization and quantization strategy:
        As the action space is tested to be within [-1, 1] in all dimensions,
        we directly scale the action to [0, 1000] using the formula:
        norm_action = (action + 1.0) / 2.0
        quantized_action = round(norm_action * action_quantization_bins)
    To invert:
        dequantized_action = (quantized_action / action_quantization_bins) * 2 - 1.0
    '''

    # --- 处理并写入数据 ---
    print("正在处理数据并生成 JSONL 文件和图像...")
    total_steps = 0
    jsonl_path = output_path / "train.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f, tqdm(desc="处理并写入数据") as pbar:
        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                for step in episode["steps"].as_numpy_iterator():
                    instruction = step["language_instruction"].decode("utf-8")
                    action = step["action"]

                    # 1. 保存图像
                    main_img_arr = step["observation"]["image"]
                    wrist_img_arr = step["observation"]["wrist_image"]

                    main_img_path = images_path / f"{total_steps:08d}_main.jpg"
                    wrist_img_path = images_path / f"{total_steps:08d}_wrist.jpg"

                    Image.fromarray(main_img_arr).save(main_img_path)
                    Image.fromarray(wrist_img_arr).save(wrist_img_path)

                    # 2. 归一化和量化动作
                    # 添加一个小的 epsilon 防止除以零
                    # norm_action = (action - action_min) / (action_max - action_min + 1e-8)

                    # directly scale action to [0, 2000]
                    norm_action = (action + np.ones(action.shape)) / 2.0  # 假设动作在 [-1, 1] 范围内
                    quantized_action = np.round(norm_action * action_quantization_bins).astype(int)
                    # 确保值在 [0, B] 范围内
                    quantized_action = np.clip(quantized_action, 0, action_quantization_bins)
                    
                    action_str = " ".join(map(str, quantized_action))

                    # 3. 构建 JSON 对象
                    prompt_text = PROMPT_TEMPLATE.format(instruction=instruction)
                    
                    data_entry = {
                        "system": system_prompt,
                        "prompt": prompt_text,
                        "response": action_str,
                        # 使用相对路径，方便数据集移动
                        "image": [
                            str(main_img_path.relative_to(output_path)),
                            str(wrist_img_path.relative_to(output_path)),
                        ],
                    }

                    # 4. 写入文件
                    f.write(json.dumps(data_entry) + "\n")

                    
                    total_steps += 1
                    pbar.update(1)

    print("\n数据转换完成！")
    print(f"总共处理了 {total_steps} 个时间步。")
    print(f"JSONL 文件保存在: {jsonl_path}")
    print(f"所有图像保存在: {images_path}")


if __name__ == "__main__":
    # 使用 tyro 解析命令行参数，方便调用
    tyro.cli(main)