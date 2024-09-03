# LLM实验文档

## 目录

- [环境准备](#环境准备)
- [准备实验配置](#准备实验配置)
- [运行实验](#运行试验)
- [收集实验结果](#收集试验结果)

## 环境准备

SWIFT支持了exp（实验）能力，该能力是为了将多个需要进行的对比实验方便地进行管理。实验能力包含的主要功能有：

- 支持在单机多卡（单机单卡下）并行运行多个训练（导出）等任务，并将超参数、训练输出、训练指标等信息记录下来，显卡占满情况下会排队
- 支持直接运行训练（或导出）后的评测任务，并将评测指标记录下来
- 支持将所有的指标生成MarkDown格式的表格方便对比
- 支持重复幂等运行，已完成实验不会重复运行

该能力是对SWIFT训练、推理、评测能力的补充，本质是多个任务的调度能力。

## 准备实验配置

一个示例实验配置如下：

```json
{
    "cmd": "sft",
    "requirements":{
        "gpu": "1",
        "ddp": "1"
    },
    "eval_requirements": {
      "gpu": "1"
    },
    "eval_dataset": ["ceval", "gsm8k", "arc"],
    "args": {
      "model_type": "qwen-7b-chat",
      "dataset": "ms-agent",
      "train_dataset_mix_ratio": 2.0,
      "batch_size": 1,
      "max_length": 2048,
      "use_loss_scale": true,
      "gradient_accumulation_steps": 16,
      "learning_rate": 5e-5,
      "use_flash_attn": true,
      "eval_steps": 2000,
      "save_steps": 2000,
      "train_dataset_sample": -1,
      "val_dataset_sample": 5000,
      "num_train_epochs": 2,
      "check_dataset_strategy": "none",
      "gradient_checkpointing": true,
      "weight_decay": 0.01,
      "warmup_ratio": 0.03,
      "save_total_limit": 2,
      "logging_steps": 10
    },
    "experiment": [
      {
        "name": "lora",
        "args": {
          "sft_type": "lora",
          "lora_target_modules": "ALL",
          "lora_rank": 8,
          "lora_alpha": 32
        }
      },
      {
        "name": "lora+",
        "args": {
          "sft_type": "lora",
          "lora_target_modules": "ALL",
          "lora_rank": 8,
          "lora_alpha": 32,
          "lora_lr_ratio": 16.0
        }
      }
    ]
}
```

- cmd：本实验运行的swift命令
- requirements：配置gpu数量和ddp数量
- eval_requirements：评测使用的gpu数量
- eval_dataset：评测使用的数据集，如果不配置则不进行评测
- args：cmd命令对应的参数
- experiment：每个子实验的独立参数，会覆盖上面的参数。必须包含name字段以存储实验结果

可以查看[这个文件夹](https://github.com/modelscope/swift/tree/main/scripts/benchmark/config)获取当前已经配置的实验示例。

## 运行实验

```shell
# 在swift根目录下运行
PYTHONPATH=. nohup python scripts/benchmark/exp.py --save_dir './experiment' --config your-config-path > run.log 2>&1 &
```

--config参数支持一个实验配置文件或一个文件夹，当指定文件夹时会并行运行其内所有的实验配置。

运行试验后会讲每个实验的日志单独记录在`./exp`文件夹内，实验结果会记录在`--save_dir`指定的文件夹内

## 收集实验结果

```shell
# 在swift根目录下运行
python scripts/benchmark/generate_report.py
```

实验结果的日志如下：

```text
=================Printing the sft cmd result of exp tuner==================


| exp_name | model_type | dataset | ms-bench mix ratio | tuner | tuner_params | trainable params(M) | flash_attn | gradient_checkpointing | hypers | memory | train speed(samples/s) | infer speed(tokens/s) | train_loss | eval_loss | gsm8k weighted acc | arc weighted acc | ceval weighted acc |
| -------- | ---------- | ------- | -------------------| ----- | ------------ | ------------------- | -----------| ---------------------- | ------ | ------ | ---------------------- | --------------------- | ---------- | --------- | ------------------ | ---------------- | ------------------ |
|adalora|qwen-7b-chat|ms-agent|2.0|adalora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=False|26.8389(0.3464%)|True|True|lr=5e-05/epoch=2|32.55GiB|0.92(87543 samples/95338.71 seconds)|17.33(2345 tokens/135.29 seconds)|0.57|1.07|0.391|0.665|0.569|
|adapter|qwen-7b-chat|ms-agent|2.0|adapter||33.6896(0.4344%)|True|True|lr=5e-05/epoch=2|32.19GiB|1.48(87543 samples/59067.71 seconds)|26.63(4019 tokens/150.90 seconds)|0.55|1.03|0.438|0.662|0.565|
|dora|qwen-7b-chat|ms-agent|2.0|lora|rank=8/target=ALL/alpha=32/lr_ratio=None/use_rslora=False/use_dora=True|19.2512(0.2487%)|True|True|lr=5e-05/epoch=2|32.46GiB|0.51(87543 samples/171110.54 seconds)|4.29(2413 tokens/562.32 seconds)|0.53|1.01|0.466|0.683|**0.577**|
|full+galore128|qwen-7b-chat|ms-agent|2.0|full|galore_rank=128/galore_per_parameter=false/galore_with_embedding=false|7721.3245(100.0000%)|True|True|lr=5e-05/epoch=2|47.02GiB|1.10(87543 samples/79481.96 seconds)|28.96(2400 tokens/82.88 seconds)|0.55|1.00|0.358|**0.688**|**0.577**|
...
```

可以将表格拷贝进其它文档中用于分析。
