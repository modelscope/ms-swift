# v5 best-of-n scoring over a math dataset with TWO heterogeneous reward channels.
#
#   PRM channel -- a separate process-reward MODEL (Qwen/Qwen2.5-Math-PRM-7B) passed as an item of --prm.
#                  It keeps its own weights resident on GPU.
#   ORM channel -- a pure FUNCTION (accuracy = MathAccuracy, backed by math_verify) passed as an item of
#                  --orm. It scores the sampled text and needs no GPU.
#
# A GPU-resident reward model needs its own Ray DeviceGroup, so mode=ray is REQUIRED (run_infer raises
# otherwise). plan_sampling_device_groups sizes every role off nproc_per_node: with nproc_per_node=2 the
# sampler gets GPUs [0,1] and the PRM group gets [2,3] -- 4 GPUs total. Inside the sampler's 2 GPUs,
# tp=pp=cp=1 so dp_size = 2 // 1 = 2: the sampler runs data-parallel across [0,1].
#
# --num_samples 8 draws a best-of-8 group per prompt; --output_format all stores every candidate plus
# each channel's score per row (the ranking-only knobs --reward_threshold/--n_best_to_keep apply to
# 'dpo', not 'all', so they are left unset here).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --sampler vllm \
    --dataset 'AI-MO/NuminaMath-TIR#500' \
    --num_samples 8 \
    --output_format all \
    --orm accuracy \
    --prm Qwen/Qwen2.5-Math-PRM-7B \
    --mode ray \
    --nproc_per_node 2 \
    --vllm_gpu_memory_utilization 0.85 \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --result_path ./output/math_prm_orm.jsonl
