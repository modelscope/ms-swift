CUDA_VISIBLE_DEVICES=0,1,2,3 \
INFONCE_TEMPERATURE=0.1 \
INFONCE_MASK_FAKE_NEGATIVE=true \
INFONCE_INCLUDE_QQ=true \
INFONCE_INCLUDE_DD=false \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-Embedding-0.6B \
    --task_type embedding \
    --model_type qwen3_emb \
    --train_type full \
    --dataset sentence-transformers/stsb \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 50 \
    --eval_steps 50 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-6 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --sequence_parallel_size 4 \
    --padding_free true \
    --attn_impl flash_attn
