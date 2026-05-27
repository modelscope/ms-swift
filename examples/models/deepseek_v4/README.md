# DeepSeek-V4 微调支持


目前Megatron-SWIFT支持了DeepSeek-V4的微调与RL支持，包括MTP、FP8的支持。（FP4暂时不支持，会在加载权重时自动转成FP8/BF16）

你需要使用Megatron-Core dev分支，mcore-bridge 和 ms-swift main分支。

```shell
pip install git+https://github.com/NVIDIA/Megatron-LM.git@dev
pip install git+https://github.com/modelscope/mcore-bridge.git
pip install git+https://github.com/modelscope/ms-swift.git
```

## 精度对齐

目前Megatron-Core对DeepSeek-V4的支持有算子实现有误，存在精度误差（后续可能更新），具体查看[这个issue](https://github.com/NVIDIA/Megatron-LM/issues/4957)。你需要修改部分代码：
- 修改[这行](https://github.com/NVIDIA/Megatron-LM/blob/56481b0501cf7b3719e1869c495e2680ef0f3456/megatron/core/transformer/hyper_connection.py#L76)，修改为`mixed = torch.bmm(h_res_batched.transpose(-1, -2), residual_batched).view(s, b, n, C)`
- 修改[这行](https://github.com/NVIDIA/Megatron-LM/blob/56481b0501cf7b3719e1869c495e2680ef0f3456/megatron/core/transformer/hyper_connection.py#L386)，修改为`h_res_batched = h_res.transpose(-1, -2).contiguous().view(s * b, n, n)`
- 此外为了支持精度对齐测试（FP32），你还注释掉修改[这几行](https://github.com/NVIDIA/Megatron-LM/blob/56481b0501cf7b3719e1869c495e2680ef0f3456/megatron/core/transformer/experimental_attention_variant/dsa.py#L41-L43)。

修改完代码后，测试以下代码，确认无实现错误（测试transformers/megatron forward对齐情况）：

创建mini版本的模型，我们将创建4层：

```

```




## LoRA训练



```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
              'AI-ModelScope/alpaca-gpt4-data-en#1000' \
              'swift/self-cognition#1000' \
    --model_author swift \
    --model_name swift-robot \
    --merge_lora true \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --micro_batch_size 4 \
    --global_batch_size 32 \
    --padding_free false \
    --group_by_length true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/DeepSeek-V4-Flash \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --mtp_num_layers 1 \
    --attention_backend flash
```

显存占用：
![显存占用](./asset/memory.png)


损失函数：
![loss](./asset/loss.png)

提示：
- 如果你要设置pp并行，你需要额外设置。例如：
```
--pipeline_model_parallel_size 2 \
--pipeline_model_parallel_layout 'Et*22|t*21mL' \
```
- 全参数训练也是支持的，你需要降低learning_rate，并提高并行数。参考64卡训练例子：
```
--lr 1e-5 \
--min_lr 1e-6 \
--tensor_model_parallel_size 1 \
--expert_model_parallel_size 8 \
--pipeline_model_parallel_size 8 \
--pipeline_model_parallel_layout Et*5|t*5|t*6|t*6|t*6|t*5|t*5|t*5mL \
```
- 暂时不支持`paddind_free`和`packing`，但可以通过`group_by_length`加速。暂时不支持TP，待Megatron-Core支持。
- FP8训练：你可以设置以下参数开启FP8训练，并最终将权重保存成FP8权重。推荐使用全参数。如果要使用LoRA + FP8，你需要只保存LoRA权重，并使用BF16权重进行Merge-LoRA（FP8 精度有限，LoRA delta 会被舍入为 0）。参考[这个例子](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/fp8/lora.sh)。
```
--fp8_recipe blockwise \
--fp8_format e4m3 \
--fp8_param_gather true \
``

推理训练后的模型：

```shell

```

训练结果：

