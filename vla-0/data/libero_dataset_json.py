# convert_libero_to_vla_json_format.py
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

# --- VLA 格式化模板 ---
# 新的系统提示，指导模型输出 JSON 格式的动作
SYSTEM_PROMPT_TEMPLATE = (
    "Assume you are a robot control system. "
    "Analyze the input image, the task description and the current state to predict the robot's next action. "
    "The output should be a sequence of actions where"
    "the first three values represent 'dx', 'dy', 'dz' for the end-effector's positional movement, "
    "the next three values represent 'drx', 'dry', 'drz' for the end-effector's rotational movement, "
    "the seventh value indicates the gripper state (-1: open, 1: closed). "
    "Provide only next action as output, nothing else."
)

# 包含两个图像占位符
PROMPT_TEMPLATE = "robot state: '{robot_state}'. task description: {instruction}. image:<image><image>"


def main(
    data_dir: str = "/home/yuquan002/ssd/modified_libero_rlds",
    output_dir: str = "/home/yuquan002/ssd/libero_vl_dataset",
):
    """
    将 LIBERO RLDS 数据集转换为适用于 VLM 微调的 JSONL 格式。
    每个时间步的动作被格式化为一个 JSON 对象。

    Args:
        data_dir: 包含原始 LIBERO RLDS 数据集的目录。
        output_dir: 输出转换后数据的目录。
    """
    output_path = pathlib.Path(output_dir) / RAW_DATASET_NAMES[0].replace("no_noops", "raw")
    images_path = output_path / "images"

    # 清理并创建输出目录
    print(f"将数据输出到: {output_path}")
    if output_path.exists():
        print("警告: 输出目录已存在，将进行覆盖。")
        shutil.rmtree(output_path)
    images_path.mkdir(parents=True, exist_ok=True)
    
    # 设置固定的系统提示
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    print("系统提示:")
    print(system_prompt)
    print("-" * 50)

    # --- 处理并写入数据 ---
    print("正在处理数据并生成 JSONL 文件和图像...")
    total_steps = 0
    jsonl_path = output_path / "libero.jsonl"
    
    with open(jsonl_path, "w", encoding="utf-8") as f, tqdm(desc="处理并写入数据") as pbar:
        for raw_dataset_name in RAW_DATASET_NAMES:
            # 加载原始数据集
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                for step in episode["steps"].as_numpy_iterator():
                    # a. 提取所需数据
                    instruction = step["language_instruction"].decode("utf-8")
                    action = step["action"]
                    robot_state = step["observation"]["state"]
                    # main_img_arr = step["observation"]["image"]
                    # wrist_img_arr = step["observation"]["wrist_image"]

                    # b. 保存图像
                    main_img_path = images_path / f"{total_steps:08d}_main.jpg"
                    wrist_img_path = images_path / f"{total_steps:08d}_wrist.jpg"
                    # Image.fromarray(main_img_arr).save(main_img_path)
                    # Image.fromarray(wrist_img_arr).save(wrist_img_path)

                    # c. 格式化输入 (Prompt)
                    # 将机器人状态数组转换为紧凑的字符串格式，例如 '[0.1, -0.2, ...]'
                    # 将夹爪状态二值化, 1.0 表示闭合, -1.0 表示打开
                    # 格式上转为7维，与输出动作对应
                    robot_state_7dim = np.zeros((7,), dtype=np.float32)
                    robot_state_7dim[0:3] = robot_state[0:3] * 100.0
                    robot_state_7dim[3:6] = robot_state[3:6]
                    if robot_state[-1] > -0.01:
                        robot_state_7dim[6] = 1.0  # gripper closed
                    else:
                        robot_state_7dim[6] = -1.0  # gripper open

                    robot_state_str = np.array2string(robot_state_7dim, precision=3, separator=', ', suppress_small=True)

                    prompt_text = PROMPT_TEMPLATE.format(
                        robot_state=robot_state_str,
                        instruction=instruction
                    )
                    # convert action to text
                    action_text = np.array2string(action, precision=3, separator=', ', suppress_small=True)

                    # d. 格式化输出 (Response)
                    # 将动作text转换为 JSON 字符串
                    # 这是模型需要学习生成的直接目标
                    # action_json_str = json.dumps(action_text)

                    # e. 构建最终的 JSONL 数据条目
                    data_entry = {
                        "system": system_prompt,
                        "prompt": prompt_text,
                        "response": action_text,
                        # 使用相对路径，方便数据集移动
                        "image": [
                            str(main_img_path.relative_to(output_path)),
                            str(wrist_img_path.relative_to(output_path)),
                        ],
                    }
                    # import pdb; pdb.set_trace()

                    # f. 写入文件
                    f.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                    
                    total_steps += 1
                    pbar.update(1)

    print("\n数据转换完成！")
    print(f"总共处理了 {total_steps} 个时间步。")
    print(f"JSONL 文件保存在: {jsonl_path}")
    print(f"所有图像保存在: {images_path}")

if __name__ == "__main__":
    # 使用 tyro 解析命令行参数，方便调用
    tyro.cli(main)