# ref: https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E8%87%AA%E6%88%91%E8%AE%A4%E7%9F%A5%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md
# Experimental environment: A100
# 70GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path SUSTC/SUS-Chat-34B \
    --dataset alpaca-zh alpaca-en \
    --train_dataset_sample 500 \
    --eval_steps 20 \
    --logging_steps 5 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
