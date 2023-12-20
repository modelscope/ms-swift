# Experimental environment: V100, A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "/mnt/workspace/yzhao/tastelikefeet/swift/examples/pytorch/llm/output/cogagent-chat/v47-20231220-132558/checkpoint-400" \
    --load_args_from_ckpt_dir true \
    --eval_human true \
    --max_length 4096 \
    --use_flash_attn true \
    --max_new_tokens 2048 \
    --temperature 0.3 \
    --top_p 0.7 \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false \
