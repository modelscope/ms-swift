## LoRA Training

Best practice reference for single-node 8xH20 LoRA training with Qwen3-235B-A22B-Instruct-250718: https://github.com/modelscope/ms-swift/pull/5033.

Compared to full parameter tuning, LoRA training differs in both the training and MCore-to-HF conversion scripts:

Training Script:

```bash
# full: 2 * 70GiB 0.61s/it
# lora: 2 * 14GiB 0.45s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```
- For LoRA training scripts of MoE models, please refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/lora).

MCore to HF Conversion Script:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```

- Note: The `mcore_adapters` folder contains an `args.json` file. During the conversion process, parameters related to `mcore_model` and LoRA will be loaded from this file. The system will then perform a merge-lora operation between the `mcore_model` and `mcore_adapters` to obtain the complete model weights, and finally convert them into HuggingFace (HF) format.

If you only want to merge the LoRA weights without converting them to Hugging Face format, for subsequent DPO training, you can use the following script:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-mcore \
    --test_convert_precision true
```


## Benchmark
The speed comparison of full-parameter training for Dense/MoE models using `megatron sft` and `swift sft` on a single machine with eight A800 GPUs is shown below. The corresponding scripts can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/benchmark).

**Dense** Qwen2.5-14B:


|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 9.04s/it    | 10.32s/it       | 10.56s/it       |
| GPU Memory Usage | 8\*64GB      | 8\*80GB          | 8\*58GB          |

**MoE** Qwen1.5-MoE-A2.7B:

|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 2.95s/it    | 6.02s/it        | 24.30s/it       |
| GPU Memory Usage | 8\*57GB      | 8\*72GB          | 8\*50GB          |
