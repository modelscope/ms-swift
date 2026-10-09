# v5 base -> chat: turn a BASE (pretrained, non-instruct) model into a chat model with full-parameter SFT.
#
# Qwen/Qwen2.5-0.5B is the raw pretrained base -- it continues text but does not follow the turn-taking chat
# format. This example fine-tunes it on conversational data WITH a chat template applied, so it learns the
# system/user/assistant structure and to emit an end-of-turn. The mechanics that make it "base to chat":
#   --model points at the BASE checkpoint (not -Instruct),
#   --use_chat_template true wraps every row in the model family's chat template (the base repo ships no
#     chat config, so this is what imposes the conversational format),
#   --tuner full updates all weights -- a base model usually needs full-parameter SFT (not a small adapter) to
#     actually acquire instruction-following, since there is no prior chat behaviour to nudge.
#
# DATASET: self-cognition is used here only because it is small and cached, so the smoke is fast; it teaches
# the chat FORMAT plus a narrow identity. For a real base->chat conversion point --dataset at a broad
# instruction corpus (e.g. an alpaca-style set) so the model learns general following, not one persona.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B \
    --task_type causal_lm \
    --use_chat_template true \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
