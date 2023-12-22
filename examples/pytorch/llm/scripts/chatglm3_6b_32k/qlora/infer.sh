# Experimental environment: A10, 3090
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --ckpt_dir "/mnt/workspace/yzhao/tastelikefeet/swift/examples/pytorch/llm/output/chatglm3-6b-32k/v12-20231111-230758/checkpoint-300" \
    --load_dataset_config true \
    --max_length 4096 \
    --max_new_tokens 2048 \
    --temperature 0.1 \
    --top_p 0.7 \
    --eval_human true \
    --repetition_penalty 1.05 \
    --do_sample true \
    --merge_lora_and_save false \
