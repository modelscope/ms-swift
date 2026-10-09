# v5 GRPO with MoE ROUTING REPLAY (R2/R3) -- a feature sub-example of ../grpo.sh, for a Qwen3-MoE policy.
#
# USE_SWIFT_V5=1 routes `swift rl` to the v5 dev CLI. `--rlhf_type grpo` dispatches to
# swift.dev.recipe.run_grpo. Read ../grpo.sh FIRST: this file changes only how the MoE router is handled in
# the training forward, not the GRPO objective. Everything GRPO means is explained there.
#
# The problem routing replay solves: an MoE router picks each token's experts from the CURRENT weights. In
# GRPO the policy moves between the rollout (which generated a token) and the training forward (which
# recomputes that token's log-prob for the importance ratio). A training forward that RE-ROUTES under the
# drifted weights can select DIFFERENT experts than the ones that actually generated the token, so the
# recomputed log-prob -- and thus the GRPO ratio -- is measured against the wrong expert mixture. Routing
# replay fixes this: the training forward is driven with `router_replay_action='replay_forward'` and each
# row's recorded `routed_experts`, so it REPLAYS the generating routing instead of recomputing it.
#
# Two modes differ only in WHERE the replayed routing comes from (then feed the same replay path):
#   - R2  RECORD the training model's OWN routing for the rollout tokens (an extra `forward_only` under the
#         just-synced generating weights -- a faithful proxy), then replay it. Use R2 when the sampler does
#         NOT export routing. Costs one extra full-sequence forward per window. This is the default below.
#   - R3  replay the routing the SAMPLER exported at generation time. Needs a vLLM build + MoE checkpoint
#         that report routing, turned on with `--vllm_engine_kwargs enable_return_routed_experts=true`; the
#         installed vLLM exports it for Qwen3-MoE. R3 avoids R2's extra forward (the routing rides the
#         generation), but if the engine exports none the loop RAISES ('did not export') rather than silently
#         recomputing -- switch to R2 in that case.
#
# Hard constraints (validate refuses the illegal ones up front; the model/engine ones are loud runtime errors):
#   - GRPO-ONLY. R2/R3 replay is driven inside GRPOLoop; the RFT/GKD/OPSD/PPO loops build their own
#     forward_backward kwargs without it, so `--router_replay_mode` on a non-GRPO run is rejected.
#   - MoE-ONLY. A dense policy has no expert routing to record, so R2 on a dense model raises at runtime.
#   - CHORD-INCOMPATIBLE. CHORD's auxiliary SFT rows carry no expert routing, so they cannot share the replay
#     forward; `--chord_sft_dataset` + routing replay is rejected.
#   - Backend-agnostic: R2/R3 work on BOTH `--backend hf` (transformers, default here) and `--backend
#     megatron` (mcore's moe_enable_routing_replay). Nothing here is transformers-specific.
#
# LoRA target scoping (why --target_modules names the attention projections, NOT all-linear): on Qwen3-MoE the
# experts are PACKED 3D parameters, so peft resolves `all-linear` to the attention projections PLUS the expert
# weights (as target_parameters). With the experts wrapped, the transformers R2 RECORD setup can no longer read
# `top_k` off the MoE block and raises. Naming q/k/v/o_proj keeps the LoRA off the packed experts so the
# router gate stays native and the RECORD->REPLAY mechanism runs cleanly. (On `--backend megatron` LoRA is
# built via mcore-bridge, not peft, so `all-linear` is fine there.)
#
# HOW TO READ THIS RUN: exactly as ../grpo.sh -- judge convergence by the REWARD curve (`step N loss= reward=`,
# tensorboard train/total_reward climbing). Routing replay adds NO new tensorboard tags; its effect is that the
# GRPO ratio (train/clip_ratio, train/approx_kl) is computed against the CORRECT expert mixture, so an MoE run
# stays honest as the policy drifts. A green run that trains lora_B is itself the evidence the RECORD->REPLAY
# path executed: R2's RECORD is fail-loudly (a dense policy or dead wiring returns no routing and raises).
#
# DEVICES: Qwen3-30B-A3B is a 30B-total / 3B-active MoE (~60GB bf16). The default below runs colocate over 4
# cards; pick FREE cards (this box keeps GPU 0 busy). Adjust CUDA_VISIBLE_DEVICES, --nproc_per_node and
# --vllm_gpu_memory_utilization to the cards/memory you actually have.
#
# This is a real training scaffold. For a fast smoke run shrink the bounded dimensions only: a tiny slice
# (`#64`), --max_steps 4, a short --max_completion_length, and fewer --nproc_per_node. Keep
# --router_replay_mode and the attention-only --target_modules -- they are the legality-defining flags; the
# MoE model itself CANNOT be shrunk (the expert routing IS the feature), so this example needs a real MoE.
CUDA_VISIBLE_DEVICES=1,2,3,4 \
USE_SWIFT_V5=1 \
swift rl \
    --rlhf_type grpo \
    --model Qwen/Qwen3-30B-A3B \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --system "You are a helpful math assistant. Solve the problem step by step and put your final answer within \boxed{}." \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --torch_dtype bfloat16 \
    --orm accuracy \
    --router_replay_mode R2 \
    --nproc_per_node 4 \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --num_generations 8 \
    --advantage_estimator grpo \
    --epsilon 0.2 \
    --num_iterations 1 \
    --beta 0.04 \
    --temperature 1.0 \
    --max_completion_length 2048 \
    --max_length 3072 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --log_completions true \
    --gradient_checkpointing true \
    --output_dir output \
    --report_to tensorboard

# R3 variant: replay the SAMPLER's exported routing instead of recording it on the training model. Change
# `--router_replay_mode R2` to `R3` and ask the vLLM engine to export each MoE layer's routing:
#     --router_replay_mode R3 \
#     --vllm_engine_kwargs '{"enable_return_routed_experts": true}' \
# R3 drops R2's extra RECORD forward (the routing rides the generation). If the engine exports no routing the
# loop raises 'did not export' and points back at R2 -- use R2 with a sampler/build that does not report routing.
