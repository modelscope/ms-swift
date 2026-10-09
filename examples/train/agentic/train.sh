# v5 agentic SFT: fine-tune tool use on an agent dataset, weighted by --loss_scale.
#
# There is NO --agent_template flag in v5 (it was removed). The agent surface is expressed two ways:
#   - the chat TEMPLATE renders the conversation into the model's own format; it comes from the model's
#     registered template, so you do not select it separately here;
#   - --loss_scale picks the weighting strategy that knows the agent transcript's markers. It scores the
#     spans the model must LEARN TO EMIT as the target while down-weighting the spans it only reads.
#
# ms_agent is the canonical ReAct-format agent set: the system prompt lists the callable APIs and the
# Thought/Action/Action Input/Observation scaffolding, and each assistant turn carries those markers
# inline (there is no separate tool_call role). --loss_scale react (swift/loss_scale/config/react.json)
# up-weights `Action:` / `Action Input:` (2.0) as the learning target, zeroes the `Observation:` CONTENT
# (the tool result the model reads but never emits), and leaves `Thought:` / `Final Answer:` at 1.0.
# Other agent strategies: hermes / qwen / agentflan / alpha_umi (swift/loss_scale/mapping.py); strategies
# chain with '+' (e.g. last_round+react). For the modern function-calling paradigm -- a dataset with a
# `tools` column and discrete tool_call / tool_response roles rendered by the model's tool-call template
# -- use --loss_scale hermes instead. Slice the dataset with #N to bound a smoke run.
#
# ms_agent transcripts are long (the system prompt embeds every tool's schema, and observations carry
# full tool output), so max_length is 4096 and --truncation_strategy delete drops the rare longer row
# rather than cutting it: truncating mid-transcript would sever the Action/Observation spans react weights.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'iic/ms_agent#2000' \
    --loss_scale react \
    --truncation_strategy delete \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
