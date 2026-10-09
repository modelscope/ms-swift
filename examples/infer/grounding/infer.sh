# v5 GROUNDING (visual object detection) inference with a vision-language model.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. Grounding is plain causal_lm GENERATION over an
# image: the VL model reads the picture plus a detection prompt and emits bounding boxes as text. There is
# no special task_type -- it is the multimodal generation path.
#
# WHY --sampler vllm (not transformers): the v5 transformers sampler loads weights through HF
# AutoModelForCausalLM, which REJECTS a vision-language config (Qwen2_5_VLConfig is not a causal-LM config),
# so a VL checkpoint must run on a served engine. `--sampler vllm` (or `sglang`) loads the model with its
# ...ForConditionalGeneration class and serves the image tokens; keep grounding on vllm/sglang.
#
# DATA SHAPE (examples/infer/data/grounding.jsonl; run this script from the repo root): each row carries a
# `messages` turn whose content holds an `<image>` placeholder + the detection instruction, and a parallel
# `images` list (URL or local path) supplying that image. Ask in natural language ("Locate every X ...
# output bbox_2d and label") -- the terse "Task: Object Detection" phrasing only works with a dedicated
# grounding adapter, not a base instruct VL model.
#
# OUTPUT: `response` is the model's JSON -- a list of {"bbox_2d": [x1, y1, x2, y2], "label": ...} boxes in
# the model's normalized-1000 coordinate space. MAX_PIXELS caps the image resolution fed to the vision tower
# (lower = faster, coarser boxes).
#
# Real inference scaffold, smoke-verified end to end (Qwen2.5-VL-3B-Instruct on vLLM: the animal.png row
# returned 4 distinct animal boxes, cat.png returned 1). For a fast smoke shrink only bounded dims
# (--vllm_gpu_memory_utilization, MAX_PIXELS, --max_new_tokens, dataset slice); keep --sampler vllm.
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --sampler vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --dataset examples/infer/data/grounding.jsonl \
    --max_new_tokens 512 \
    --temperature 0 \
    --output_format all \
    --result_path ./output/grounding_infer.jsonl
