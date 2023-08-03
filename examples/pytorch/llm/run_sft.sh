CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_type qwen-7b \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000
