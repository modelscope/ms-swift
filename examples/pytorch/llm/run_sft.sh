CUDA_VISIBLE_DEVICES=0,1 \
python llm_sft.py \
    --model_type openbuddy-llama2-13b \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000
