# Test script for VLA-0 policy inferencing
import os

import test
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from server_vla0_policy import VLA0Policy
import json
import logging
import os
import pathlib
from typing import Any, Dict, Optional
import numpy as np
import torch
import tyro
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

def decode_action_text(
    action_text: str,
    quantization_bins: int = 1000,
    action_min: float = -1.0,
    action_max: float = 1.0
) -> np.ndarray:
    try:
        # 1. 将空格分隔的文本解码为整数
        quantized_action = np.fromstring(action_text, dtype=int, sep=' ')
        # 2. 反归一化
        norm_action = quantized_action / quantization_bins
        continuous_action = norm_action * (action_max - action_min) + action_min
        return continuous_action
    except (ValueError, TypeError) as e:
        logging.error(f"无法解析动作文本: '{action_text}'. 错误: {e}")
        # 返回一个安全的零动作或根据需要处理
        num_dims = 7 # 假设动作维度为7，可以根据需要修改
        return np.zeros(num_dims)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "/home/yuquan002/ssd/ms-swift-robotics/output/qwen3-vl-4b-instruct-vla0-libero/v9-20251107-182735/checkpoint-4096"

    logging.info(f"Loading model from {model_path} to device: {device}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    logging.info(f"Model loaded successfully.")
    processor = AutoProcessor.from_pretrained(model_path)

    policy = VLA0Policy(model, processor, device)

    # get prompt input from dataset
    dataset_path = '/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla/train_1k.jsonl'
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]

    for test_sample in test_data:

        system_message = test_sample['system']
        query_message = test_sample['prompt']
        images = test_sample['image']
        
        main_image_path = os.path.join('/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla', images[0])
        wrist_image_path = os.path.join('/home/yuquan002/ssd/libero_vl_dataset/libero_spatial_vla', images[1])
        main_image = Image.open(main_image_path).convert("RGB")
        wrist_image = Image.open(wrist_image_path).convert("RGB")

        obs = {
            "observation/image": np.array(main_image),
            "observation/wrist_image": np.array(wrist_image),
            "prompt": query_message
        }

        action_floats = [float(x) for x in test_sample['response'].strip('[]').split(',')]
        test_action = np.array(action_floats)

        action = policy.infer(obs)
        print("Predicted Action Text:", action)
        print("Expected Action Text:", test_sample['response'])
        # print("Expected Action Text:", test_action)

        # calculate error
        diff = action - test_action
        print("Difference:", diff)
        error = np.linalg.norm(action - test_action)
        print("Prediction Error (L2 norm):", error)

        # press any key to continue
        input("Press Enter to continue to the next sample...")
        


