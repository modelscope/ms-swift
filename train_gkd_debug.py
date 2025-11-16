#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GKD Training Script - Debuggable Version
可以直接在IDE中运行和调试,无需使用shell脚本

使用方法:
1. 单GPU调试 (不使用DeepSpeed):
   python train_gkd_debug.py

2. 单GPU使用DeepSpeed:
   torchrun --nproc_per_node=1 train_gkd_debug.py

3. 多GPU:
   torchrun --nproc_per_node=N train_gkd_debug.py
"""
import os
from swift.llm import rlhf_main, RLHFArguments


def main():
    # 设置环境变量
    os.environ['WANDB_API_KEY'] = '28e11ef52849c4640b93051377be27eafac62c44'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 创建参数对象
    args = RLHFArguments(
        # RLHF类型和模型配置
        rlhf_type='gkd',
        model='Qwen/Qwen2.5-0.5B-Instruct',

        # Teacher模型配置
        external_plugins=['rp_teacher_adapter.py'],
        teacher_adapter='rp_teacher_adapter',
        teacher_model='Qwen/Qwen2.5-0.5B-Instruct',
        # 调试模式: 关闭teacher的DeepSpeed
        teacher_deepspeed=None,  # 改为 'zero3' 启用DeepSpeed
        # teacher_deepspeed='zero3',  # 启用DeepSpeed时取消注释

        # 训练类型
        train_type='full',

        # 数据集配置
        dataset=[
            'processed_training_data_final_fixed.jsonl',
            'benchmark_datasets_filtered_14k/alignbench_v1.1.jsonl',
            'benchmark_datasets_filtered_14k/arena_hard.jsonl',
            'benchmark_datasets_filtered_14k/arena_multi_turn_10-20.jsonl',
            'benchmark_datasets_filtered_14k/creative_writing_v3.jsonl',
            'benchmark_datasets_filtered_14k/ifeval.jsonl',
            'benchmark_datasets_filtered_14k/wildchat_gpt4_10-40.jsonl',
            'benchmark_datasets_filtered_14k/writingbench.jsonl',
        ],

        # GKD特定参数
        seq_kd=False,
        lmbda=1,
        beta=1,

        # 长度和截断配置
        truncation_strategy='delete',
        max_length=17000,
        max_model_len=17000,
        max_completion_length=200,

        # 训练超参数
        torch_dtype='bfloat16',
        num_train_epochs=2,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,

        # 保存和日志配置
        save_steps=500,
        save_total_limit=8,
        logging_steps=1,
        output_dir='condition_distill',
        save_only_model=True,

        # 数据加载配置
        dataloader_num_workers=64,
        dataset_num_proc=4,

        # DeepSpeed配置
        # 调试模式: 关闭DeepSpeed以避免分布式训练的复杂性
        # 如需使用DeepSpeed,请用: torchrun --nproc_per_node=1 train_gkd_debug.py
        deepspeed=None,  # 改为 'zero3' 启用DeepSpeed
        # deepspeed='zero3',  # 启用DeepSpeed时取消注释

        # 注意力实现
        attn_impl='flash_attn',

        # vLLM配置
        use_vllm=True,
        vllm_mode='server',
        vllm_server_host=['127.0.0.1'],  # 必须是列表
        vllm_server_port=[8001],  # 必须是列表

        # 其他配置
        report_to=['wandb'],
        use_hf=True,
    )

    # 运行训练
    # 在这里设置断点即可调试
    rlhf_main(args)


if __name__ == '__main__':
    main()
