## 复用 hf 的 ppo trainer
# generate ? rollout 还是 GEN rm 采用的是 hf 框架，类似llamafactory

export WANDB_MODE=offline
nproc_per_node=8
# MAX_PIXELS=409600 for VL
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=$nproc_per_node
swift rlhf
--rlhf_type safe_rlhf_v_ppo
--model 
--reward_model /zhoupc/safe_alignment/models/safe_rlhf_v/rm_qwen2_5_vl
--cost_model /zhoupc/safe_alignment/models/safe_rlhf_v/cm_qwen2_5_vl # 
--train_type full
--dataset /zhoupc/safe_alignment/datasets/converted_sample.jsonl
--torch_dtype bfloat16
--num_train_epochs 2
--per_device_train_batch_size 1
--per_device_eval_batch_size 1
--attn_impl flash_attn
--learning_rate 5e-7
--remove_unused_columns false #?
--warmup_ratio 0.03
--dataloader_num_workers 0
--deepspeed zero3_offload 
--dataset_num_proc 8 

--freeze_vit true

--gradient_accumulation_steps 4
--eval_steps 3000
--save_steps 10000
--save_total_limit 1
--logging_steps 5 
--max_length 21000
## Saving settings
--save_only_model true
--output_dir /zhoupc/safe_alignment/checkpoints/safe_rlhf_v_ppo_qwen-7b
## TRL PPO settings
--lambda_init 10 
--lambda_max 20
--lambda_lr 0.1
--lambda_update_delay_steps 1
--episode_cost_window_size 128
--threshold -0.5
--kl_coef 0.02
--gamma 1.0 # 折扣因子
--lam 0.95 # 优势函数衰减因子
--cliprange_value 5.0 # 计算损失更新权重时，为了避免单步更新过大，对上一步的价值函数值，添加 该范围，作为下一步的取值范围
--cliprange 0.2 # 裁剪策略损失到 1-ε, 1+ε 范围内
--cliprange_score 50.0
## Rollout settings?
--top_p 1.0
--temperature 1.0
--report_to wandb
