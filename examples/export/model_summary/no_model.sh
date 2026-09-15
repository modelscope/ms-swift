#!/bin/bash
# 只查看 tokenizer 与 template，不实例化模型（model 为 None）。
#
# 原理：
#   --load_model false  跳过 get_model()，只准备 config + processor(tokenizer)，
#                       权重文件同样不会被下载。
#
# 适用场景：调试对话模板、检查 tokenize 结果、确认 max_length / special tokens。
# 对 Qwen3 这类纯文本模型（template.use_model = False），这个模式已经够用，
# 不需要 dummy model。

set -e

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen3-tokenizer-summary}

export CUDA_VISIBLE_DEVICES=""

swift export \
    --model "$MODEL" \
    --to_model_summary true \
    --load_model false \
    --output_dir "$OUTPUT_DIR" \
    --exist_ok true
