# v5 REWARD MODEL inference: score a whole conversation with one scalar reward.
#
# USE_SWIFT_V5=1 routes `swift infer` to the v5 dev CLI. A reward model is NOT a distinct task_type -- it is
# a sequence classifier with a single output: `--task_type seq_cls --num_labels 1`. (`swift rl --rlhf_type rm`
# derives exactly these two when it TRAINS a reward head; at inference there is no rlhf_config, so you state
# them yourself.) `--sampler transformers` forwards each conversation through HF's classification head; on a
# pooling engine (`--sampler vllm`/`sglang`) the same command runs through the engine's `encode`.
#
# DATA SHAPE (examples/infer/data/reward_model.jsonl; run this script from the repo root): a full
# conversation per row -- messages = [user, assistant, ...]. The reward is scored over the whole exchange
# (there is no separate "response" to generate); the assistant turn is part of the input, not a target.
#
# OUTPUT: each row's `response` is a 1-wide list holding the scalar reward (num_labels=1); `responses` wraps
# it. Higher = the reward model prefers that conversation.
#
# *** THE HEAD MUST BE TRAINED ***: pointing --model at a plain instruct/base checkpoint builds a
# randomly-initialized `score` head (the load report prints `score.weight ... MISSING`), so the numbers are
# meaningless -- they only prove the wiring. For real rewards point at an actual reward model: either
# --adapters at the LoRA/merges you trained with `swift rl --rlhf_type rm`, or --model at a trained RM
# checkpoint. This example uses a Qwen backbone to exercise the seq_cls/num_labels=1 path end to end.
#
# Real inference scaffold; the seq_cls+num_labels=1 reward path is smoke-verified (a smaller Qwen2.5-0.5B
# variant, identical flags). Shrink only bounded dims for a fast smoke; keep --task_type/--num_labels.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --task_type seq_cls \
    --num_labels 1 \
    --sampler transformers \
    --dataset examples/infer/data/reward_model.jsonl \
    --max_batch_size 8 \
    --output_format all \
    --result_path ./output/reward_model_infer.jsonl
