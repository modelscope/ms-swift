# convert_libero_for_hierarchical_vla.py
import json
import os
import pathlib
import shutil
from typing import Dict, List

import numpy as np
import tensorflow_datasets as tfds
import tyro
from PIL import Image
from tqdm import tqdm

# 定义将要合并处理的 LIBERO 数据集子集
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# --- 新的 VLM 微调模板 ---

# 1. 为 VLM (Planner) 设计的 System Prompt
VLM_SYSTEM_PROMPT = (
    "You are an expert robot controller observing the world through a wrist and a third-person camera. "
    "Your task is to issue the next short-term command to the robot in natural language. "
    "Analyze the provided images and describe the robot's next immediate action in a clear, concise command. "
    "Focus on the immediate goal, such as 'move forward and down towards the red block' or 'close the gripper slowly'."
)

# 2. VLM 的 Prompt 模板 (腕部相机在前)
VLM_PROMPT_TEMPLATE = "<image>\n<image>\nDescribe the robot's next immediate action."

def generate_instruction_from_actions(
    actions_chunk: np.ndarray, 
    translation_threshold: float = 0.01,
    gripper_threshold: float = 0.1
) -> str:
    """
    根据一个动作块(chunk)的统计数据，启发式地生成自然语言指令。

    Args:
        actions_chunk (np.ndarray): 形状为 (CHUNK_SIZE, 7) 的动作数组。
        translation_threshold (float): 判断是否发生位移的阈值。
        gripper_threshold (float): 判断夹爪是否动作的阈值。

    Returns:
        str: 生成的自然语言指令。
    """
    if actions_chunk.shape[0] == 0:
        return "hold position"

    # --- 分析位移 ---
    # 计算 x, y, z 轴的平均速度/位移
    avg_translation = np.mean(actions_chunk[:, :3], axis=0)
    move_descriptions = []
    
    # X轴: forward/backward
    if avg_translation[0] > translation_threshold:
        move_descriptions.append("forward")
    elif avg_translation[0] < -translation_threshold:
        move_descriptions.append("backward")

    # Y轴: left/right
    if avg_translation[1] > translation_threshold:
        move_descriptions.append("left")
    elif avg_translation[1] < -translation_threshold:
        move_descriptions.append("right")

    # Z轴: up/down
    if avg_translation[2] > translation_threshold:
        move_descriptions.append("up")
    elif avg_translation[2] < -translation_threshold:
        move_descriptions.append("down")

    # --- 分析夹爪 ---
    gripper_action = "hold gripper"
    # 计算夹爪状态的总体变化
    gripper_change = actions_chunk[-1, -1] - actions_chunk[0, -1]
    if gripper_change < -gripper_threshold:
        gripper_action = "close the gripper"
    elif gripper_change > gripper_threshold:
        gripper_action = "open the gripper"

    # --- 组合指令 ---
    if not move_descriptions and gripper_action == "hold gripper":
        return "hold position"
    
    if move_descriptions:
        # e.g., "move forward, up and right"
        full_move_description = "move " + ", ".join(move_descriptions)
        if gripper_action != "hold gripper":
            # e.g., "move forward and up, while closing the gripper"
            return f"{full_move_description}, while {gripper_action}"
        else:
            return full_move_description
    else:
        # e.g., "close the gripper"
        return gripper_action


def main(
    data_dir: str = "/path/to/your/modified_libero_rlds",
    output_dir: str = "/path/to/your/libero_hierarchical_dataset",
    chunk_size: int = 5,
):
    """
    将 LIBERO RLDS 数据集转换为分层 VLA 训练格式。
    1. 为 VLM Planner 生成微调数据 (指令 -> 自然语言)
    2. 为 GRU Translator 生成训练数据 (自然语言 -> 连续动作序列)

    Args:
        data_dir: 包含原始 LIBERO RLDS 数据集的目录。
        output_dir: 输出转换后数据的目录。
        chunk_size: 降采样和升采样的步长。
    """
    output_path = pathlib.Path(output_dir)
    
    # --- VLM 微调数据路径 ---
    vlm_data_path = output_path / "vlm_finetune"
    vlm_images_path = vlm_data_path / "images"

    # --- Translator 训练数据路径 ---
    translator_data_path = output_path / "translator_training"

    # 清理并创建输出目录
    print(f"将数据输出到: {output_path}")
    if output_path.exists():
        print("警告: 输出目录已存在，将进行覆盖。")
        shutil.rmtree(output_path)
    
    vlm_images_path.mkdir(parents=True, exist_ok=True)
    translator_data_path.mkdir(parents=True, exist_ok=True)

    print("VLM (Planner) 系统提示:")
    print(VLM_SYSTEM_PROMPT)
    print("-" * 30)

    total_chunks = 0
    vlm_jsonl_path = vlm_data_path / "libero_vlm_finetune.jsonl"
    translator_jsonl_path = translator_data_path / "libero_translator_training.jsonl"

    with open(vlm_jsonl_path, "w", encoding="utf-8") as vlm_f, \
         open(translator_jsonl_path, "w", encoding="utf-8") as translator_f, \
         tqdm(desc="Processing episodes") as pbar:

        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode in raw_dataset:
                episode_steps = list(episode["steps"].as_numpy_iterator())
                num_episode_steps = len(episode_steps)
                
                # 以 CHUNK_SIZE 为步长进行迭代 (降采样)
                for i in range(0, num_episode_steps, chunk_size):
                    # 确保我们有一个完整的 chunk
                    if i + chunk_size > num_episode_steps:
                        continue
                    
                    start_step = episode_steps[i]
                    
                    # 1. 提取用于 Translator 的连续动作序列 (升采样目标)
                    action_chunk = np.array([step["action"] for step in episode_steps[i : i + chunk_size]])
                    
                    # 2. 从动作序列生成自然语言指令
                    generated_instruction = generate_instruction_from_actions(action_chunk)

                    # 如果指令是"hold position"，则跳过这个数据点，因为它信息量不大
                    if generated_instruction == "hold position":
                        continue

                    # 3. 提取用于 Translator 的初始状态
                    # 状态 = 7维关节位置 + 1维夹爪位置 = 8维
                    initial_joint_pos = start_step["observation"]["joint_positions"]
                    initial_gripper_pos = start_step["observation"]["gripper_qpos"]
                    initial_state = np.concatenate([initial_joint_pos, initial_gripper_pos]).astype(np.float32)

                    # 4. 保存 VLM 微调所需的图像 (在 chunk 开始时)
                    wrist_img_arr = start_step["observation"]["wrist_image"]
                    main_img_arr = start_step["observation"]["agentview_image"] # 确认一下原始代码中的 'image' 是 'agentview_image'

                    wrist_img_path = vlm_images_path / f"{total_chunks:08d}_wrist.jpg"
                    main_img_path = vlm_images_path / f"{total_chunks:08d}_main.jpg"

                    Image.fromarray(wrist_img_arr).save(wrist_img_path)
                    Image.fromarray(main_img_arr).save(main_img_path)

                    # --- 5. 构建并写入两个数据集 ---
                    
                    # a) VLM Planner 微调数据
                    vlm_data_entry = {
                        "system": VLM_SYSTEM_PROMPT,
                        "prompt": VLM_PROMPT_TEMPLATE,
                        "response": generated_instruction,
                        "image": [
                            str(wrist_img_path.relative_to(vlm_data_path)), # 腕部相机在前
                            str(main_img_path.relative_to(vlm_data_path)),
                        ],
                    }
                    vlm_f.write(json.dumps(vlm_data_entry) + "\n")

                    # b) GRU Translator 训练数据
                    translator_data_entry = {
                        "instruction": generated_instruction,
                        "initial_state": initial_state.tolist(), # (8,)
                        "action_chunk": action_chunk.tolist(),   # (chunk_size, 7)
                    }
                    translator_f.write(json.dumps(translator_data_entry) + "\n")

                    total_chunks += 1
                pbar.update(1)

    print("\n数据转换完成！")
    print(f"总共处理了 {total_chunks} 个有效的动作块。")
    print(f"VLM 微调数据保存在: {vlm_jsonl_path}")
    print(f"Translator 训练数据保存在: {translator_jsonl_path}")
    print(f"所有图像保存在: {vlm_images_path}")


if __name__ == "__main__":
    # 使用 tyro 解析命令行参数，方便调用
    # 示例: python convert_libero_for_hierarchical_vla.py --data_dir /path/to/rlds --output_dir /path/to/output
    tyro.cli(main)