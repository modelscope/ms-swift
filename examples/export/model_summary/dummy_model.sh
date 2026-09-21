#!/bin/bash
# 查看 Qwen3 的完整模型架构，但不加载任何预训练权重。
#
# 原理：
#   --return_dummy_model true  通过 cls(config) 按 config.json 搭建完整结构，
#                              跳过 from_pretrained，因此不读权重；
#                              同时 safetensors / bin 权重文件也不会被下载。
#   --device_map meta          在 meta 设备上建图，不分配真实内存（8B 模型也是 0 内存）。
#   --load_model true          必须为 true，否则拿不到模型对象。
#
# 注意：meta 设备上的参数不能直接前向推理，如需前向请先 model.to_empty(device='cpu')。

set -e

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen3-model-summary}

# CPU 环境显式关闭 GPU，避免 device_map 自动探测
export CUDA_VISIBLE_DEVICES=""

swift export \
    --model "$MODEL" \
    --to_model_summary true \
    --load_model true \
    --return_dummy_model true \
    --device_map meta \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --exist_ok true
