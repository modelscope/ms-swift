# 8 * 60GiB; 9s/it
# For ease of use, we use moonshotai/Moonlight-16B-A3B-Instruct, which is also based on the DeepseekV3ForCausalLM architecture.
# https://modelscope.cn/models/moonshotai/Moonlight-16B-A3B-Instruct/file/view/master/config.json?status=1
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load Moonlight-16B-A3B-Instruct-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --pipeline_model_parallel_size 2 \
    --decoder_last_pipeline_num_layers 13 \
    --moe_permute_fusion true \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Moonlight-16B-A3B-Instruct \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash
