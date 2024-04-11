# LLM Experiment Documentation

## Table of Contents

- [Environment Setup](#Environment-setup)
- [Prepare Experiment Configuration](#Prepare-experiment-configuration)
- [Run Experiments](#Run-experiments)
- [Collect Experiment Results](#Collect-experiment-results)

## Environment Setup

SWIFT supports the exp (experiment) capability, which is designed to conveniently manage multiple ablation experiments that need to be conducted. The main functions of the experiment capability include:

- Support parallel execution of multiple training (export) tasks on a single machine with multiple GPUs (or a single GPU), and record information such as hyperparameters, training outputs, training metrics, etc. Tasks will be queued when the GPUs are fully occupied.
- Support directly running evaluation tasks after training (or export), and record evaluation metrics.
- Support generating a Markdown table for easy comparison of all metrics.
- Support idempotent re-runs, and completed experiments will not be re-run.

This capability complements SWIFT's training, inference, and evaluation capabilities and is essentially a task scheduling capability.

## Prepare Experiment Configuration

An example experiment configuration is as follows:

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

- cmd: The swift command to run in this experiment
- requirements: Configure the number of GPUs and the number of ddp (data parallel distributed processes)
- eval_requirements: The number of GPUs used for evaluation
- eval_dataset: The datasets used for evaluation. If not configured, no evaluation will be performed.
- args: Parameters corresponding to the cmd command
- experiment: Independent parameters for each sub-experiment, which will override the above parameters. Must include the name field to store experiment results

You can check [this folder](https://github.com/modelscope/swift/tree/main/scripts/benchmark/config) for examples of currently configured experiments.

## Run Experiments

```shell
# Run in the swift root directory
PYTHONPATH=. nohup python scripts/benchmark/exp.py --save_dir './experiment' --config your-config-path > run.log 2>&1 &
```

The --config parameter supports an experiment configuration file or a folder. When a folder is specified, all experiment configurations in that folder will be run in parallel.

After running the experiment, the log for each experiment will be recorded separately in the `./exp` folder, and the experiment results will be recorded in the folder specified by `--save_dir`.

## Collect Experiment Results

```shell
# Run in the swift root directory
python scripts/benchmark/generate_report.py
```

The experiment result logs are as follows:

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

You can copy the table into other documents for analysis.
