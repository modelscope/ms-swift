<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.2-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-Build from source-6FEBB9.svg"></a>
</p>


<p align="center">
<a href="https://modelscope.cn/home">ModelScope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>


## Features
- Supported SFT Methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
- Supported Features: quantization, DDP, model parallelism, gradient checkpointing, gradient accumulation, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
- Supported Models:
  - qwen series: [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary)
  - qwen-vl series: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary)
  - baichuan series: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary)
  - chatglm2 series: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary)
  - llama series: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
  - openbuddy series: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary)
  - internlm series: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
  - xverse series: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary)
  - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
- Supported Datasets:
  - NLP:
    - General: [alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), [alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), [damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary)
    - Coding: [code-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), [leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary)
    - Medical: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - Law: [lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - Math: [blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), [sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - Text Generation: [advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), [dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - Classification: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), [jd-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary)
    - Other: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/files), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - Multi-Modal: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
  - Custom Dataset
- Supported Templates:
  - Text Generation: default-generation, chatglm2-generation
  - Chat: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, internlm, xverse


## News
- 2023.10.12: Supported openbuddy-mistral-7b-chat model. The corresponding shell script can be found at `scripts/openbuddy_mistral_7b_chat`.
- 2023.10.7: Supported DeepSpeed ZeRO-2, enabling LoRA (not just QLoRA) to run DDP on 2*A10. The corresponding shell script can be found at `scripts/qwen_7b_chat/lora_ddp_ds/sft.sh`.
- 2023.10.4: Supported datasets in the fields of mathematics, law, SQL, and coding: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 2023.9.26: Supported xverse model series: xverse-7b, xverse-7b-chat, xverse-13b, xverse-13b-chat. The corresponding shell script can be found at `scripts/xverse_13b`.
- 2023.9.25: Supported qwen-14b model series: qwen-14b, qwen-14b-chat. The corresponding shell script can be found at `scripts/qwen_14b`, `scripts/qwen_14b_chat`.
- 2023.9.20: Supported incremental weight merging from LoRA and QLoRA training methods into base model weights, and saved the complete model weights for easy deployment by users. You can check the command-line parameter `--merge_lora_and_save` in the `infer.sh` script.
- 2023.9.18: Supported internlm-20b model series: internlm-20b, internlm-20b-chat. The corresponding shell script can be found at `scripts/internlm_20b`, `scripts/internlm_20b_chat`.
- 2023.9.12: Supported training with MP+DDP to accelerate full-parameter fine-tuning speed. The corresponding shell script can be found at `scripts/qwen_7b_chat/full_mp_ddp/sft.sh`.
- 2023.9.5: Supported training that only saves model weights without saving intermediate states such as optimizer weights required for checkpoint resumption, avoiding long checkpoint-saving times and large storage space in full-parameter fine-tuning. You can check the command-line parameter `--only_save_model` in the `sft.sh` script.
- 2023.9.5: Supported openbuddy-llama2-70b-chat model. The corresponding shell script can be found at `scripts/openbuddy_llama2_70b_chat`.
- 2023.9.3: Supported baichuan2 model series: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat. The corresponding shell script can be found at `scripts/baichuan2_7b`, `scripts/baichuan2_7b_chat`, `scripts/baichuan2_13b_chat`.


## Prepare the Environment
Experimental environment: A10, 3090, V100, A100, ...
```bash
# Installing miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Setting up a conda virtual environment
conda create --name ms-sft python=3.10
conda activate ms-sft

# Setting up a global mirror for pip and installing related Python packages (Note the matching version of CUDA)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/modelscope/swift.git
cd swift
pip install .
cd examples/pytorch/llm
pip install -r requirements.txt -U
```


## Run SFT and Inference
Performace: full(nice) > lora > qlora

Training GPU memory: qlora(low,3090) > lora > full(2*A100)

Tips:
- You can set `--gradient_checkpointing true` during training to save GPU memory, but this will slightly decrease the training speed. This is useful if you need to train LLM on consumer-grade GPU, e.g. 3090.
- If you want to push weights to the ModelScope Hub during training, you need to set `--push_to_hub true`.
- If you want to merge LoRA weights and save during inference, you need to set `--merge_lora_and_save true`.
- If you want to use quantization, you need to install `bitsandbytes` first: `pip install bitsandbytes -U`.
- If you want to use deepspeed, you need to `pip install deepspeed -U`. Using deepspeed can save GPU memory, but this may slightly decrease the training speed.
- If you are using older GPUs like V100, you need to set `--dtype fp16`, because they do not support bf16.
- qwen recommends installing [flash-attn](https://github.com/Dao-AILab/flash-attention), which will accelerate the training and inference speed and reduce GPU memory usage (A10, 3090, V100 machines do not support flash-attn).
- Below is a shell script for running `qwen_7b_chat` directly (you just need to specify `ckpt_dir` during inference to execute it smoothly). For more model scripts, you can check the `scripts` folder. If you want to customize a shell script, it is recommended to refer to the script in `scripts/qwen_7b_chat`.
```bash
# sft lora and infer qwen-7b-chat, Requires 38GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora/sft.sh
bash scripts/qwen_7b_chat/lora/infer.sh

# sft(lora+ddp) and infer qwen-7b-chat, Requires 2*38GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# sft(lora+ddp+deepspeed) and infer qwen-7b-chat, Requires 2*18GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/lora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/lora_ddp_ds/infer.sh

# sft(lora+mp+ddp) and infer qwen-7b-chat, Requires 4*15GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# sft(qlora) and infer qwen-7b-chat, Requires 10GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# sft(qlora+ddp) and infer qwen-7b-chat, Requires 2*14GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# sft(qlora+ddp+deepspeed) and infer qwen-7b-chat, Requires 2*16GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp_ds/infer.sh

# sft(full+mp) and infer qwen-7b-chat, Requires 2*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# sft(full+mp+ddp) and infer qwen-7b-chat, Requires 4*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh
```


## User Guide
### Introduction to MODEL_MAPPING (Model Expansion)
`MODEL_MAPPING` is defined in `utils/model.py` and is used to load various types of base models. If you need to **expand the models**, you can add them here. The key represents the unique ID of the model, and the value represents the model configuration. The configuration includes the following:

- `model_id`: Required. It represents the `model_id` in the ModelScope Hub or the local model directory.
- `revision`: Used to specify the version number of the model. If `model_id` is a local model directory, this parameter is ignored; otherwise, it is required.
- `get_function`: A function to get the model and tokenizer. By default, it uses `get_model_tokenizer_from_repo` to return the model and tokenizer. If you need to set `flash_attn` or patch the model code, etc., you can customize it.
- `lora_TM`: The default `lora_target_modules` used. In our settings, it is set to `qkv`.
- `template`: The default chat template used, such as chatml, baichuan, etc. If not set, the `default` chat template is used.
- `ignore_file_pattern`: Represents the file content to ignore when downloading. This parameter is passed to `snapshot_download`. For example, `r'.+\.bin$'`, `r'.+\.savetensors$'`, etc.


### Introduction to DATASET_MAPPING (Dataset Expansion)
`DATASET_MAPPING` is defined in `utils/dataset.py` and is used to load various types of data, such as single-turn instruction fine-tuning datasets, multi-turn chat datasets, multimodal datasets, etc. If you need to **expand the datasets**, you can add them here. The key represents the unique ID of the dataset, such as alpaca-en, alpaca-zh, etc. The value is the function to get the dataset. This function does not require any parameters and should return either `HfDataset` or `Tuple[HfDataset, HfDataset]`. In the first case, the dataset processing function will split a portion of the dataset as the validation set (based on the command-line hyperparameter `dataset_test_ratio`). In the second case, the two returned datasets are used as the training set and validation set, respectively. We support fine-tuning with multiple datasets. The training and validation parts of each sub-dataset are concatenated and the merged training set and validation set are returned.

The returned `HfDataset` must comply with certain conventions. In the case of instruction fine-tuning (single-turn dialogue), it should include the `query` and `response` fields, representing the user's query for instruction fine-tuning and the assistant's response, respectively. You can refer to the `alpaca-zh` dataset for more details. In the case of multi-turn dialogue, an additional `history` field is required, representing the history of the dialogue. You can refer to the `damo-agent-mini-zh` dataset for more details. If each example in the dataset has a different `system`, an additional `system` field is required. You can also refer to the `damo-agent-mini-zh` dataset for more details. We only calculate and optimize the loss for the `response` part.


### Introduction to TEMPLATE_MAPPING (Dialogue Template Expansion)
`TEMPLATE_MAPPING` is defined in `utils/preprocess.py` and is used to preprocess text information into token lists. If you need to **expand the dialogue templates**, you can add them here. The key represents the unique ID of the chat template, such as 'default', 'chatml', etc. The value represents the configuration of the chat template, including 'prefix', 'prompt', 'chat_sep', and 'suffix'. This module retrieves the complete chat template based on these four elements, enabling support for pre-training, text generation-style SFT, and various chat-type SFT. The meanings of these four configuration elements are as follows:

- `prefix`: Represents the prefix part of the chat template, usually including the system part and relevant formats, prefix tokens, BOS token, etc. We use `{{SYSTEM}}` as a placeholder for the system part.
- `prompt`: Represents a round of dialogue in the chat template. We use `{{QUERY}}` as a placeholder for the human inquiry part in each round of dialogue, and `{{ROUND}}` represents the placeholder for the current round of dialogue, counting from 1. The assistant's reply is concatenated after the `prompt`, so we did not design a placeholder for it.
- `chat_sep`: If multiple rounds of dialogue are needed, `chat_sep` serves as the separator between each round of dialogue, such as a newline, etc.
- `suffix`: Serves as the suffix part of the chat template, usually the EOS token. It is appended after the last round of dialogue. Only the response part of the last round will calculate the loss and be optimized, while the other parts will not calculate the loss.


### sft.sh Command Line Arguments
- `--model_type`: Represents the chosen model type, default is `'qwen-7b-chat'`. Available `model_type` can be checked using `MODEL_MAPPING.keys()`.
- `--sft_type`: Represents the fine-tuning method, default is `'lora'`. The possible values are: 'lora', 'full'. If you want to use lora or qlora, you need to select `--sft_type lora`. For qlora, an additional setting `--quantization_bit 4` is required. If you want to use full-parameter fine-tuning, you need to select `--sft_type full`.
- `--tuner_backend`: Represents the backend support for lora and qlora, default is `'swift'`. The possible values are: 'swift', 'peft'.
- `--template_type`: Represents the type of dialogue template used, default is `None`, which means it retrieves the template based on `model_type` from `MODEL_MAPPING`. Available `template_type` can be checked using `TEMPLATE_MAPPING.keys()` in `utils/preprocess.py`. By modifying it, you can support pretrain, text-generation-style SFT, and various chat-type SFT.
- `--output_dir`: Represents the directory for storing checkpoints, default is `'output'`. We will concatenate `model_type` and fine-tuning version number to this directory. This allows users to perform multiple comparative experiments on different models without changing the `output_dir` command-line argument.
- `--ddp_backend`: Represents the backend support for distributed training, default is `'nccl'`. The possible values are: 'nccl', 'gloo', 'mpi', 'ccl'.
- `--seed`: Global seed value, default is 42. In distributed training, to avoid each process using the same dropout, etc., we set `seed=seed+rank`.
- `--resume_from_checkpoint`: Used for resuming training from a checkpoint, default is `None`. You can set it to the path of the checkpoint, for example: `'output/qwen-7b-chat/vx_xxx/checkpoint-xxx'`, to resume training from that checkpoint.
- `--dtype`: torch_dtype when loading the base model, default is `'bf16'`. The possible values are: 'bf16', 'fp16', 'fp32'.
- `--ignore_args_error`: Whether to ignore errors raised by command-line argument mismatch, default is `False`. If you need to copy the code to a notebook for execution, you should set it to True.
- `--dataset`: Used to select the training dataset, default is `'advertise-gen-zh'`. Available datasets can be checked using `DATASET_MAPPING.keys()`. If you want to use multiple datasets for training, you can separate them using commas, for example: `alpaca-en,alpaca-zh`.
- `--dataset_split_seed`: Specifies the seed for splitting the sub-dataset into training and validation sets, default is `42`. This parameter is ignored if the sub-dataset has already been split into training and validation sets. When multiple sub-datasets are specified in `dataset` and the function for retrieving the sub-dataset does not perform the split (i.e., returns `HfDataset` instead of `Tuple[HfDataset, HfDataset]`), we need to split the sub-dataset. Finally, we concatenate the training and validation parts of these sub-datasets to generate the training and validation sets for the complete fine-tuning dataset.
- `--dataset_test_ratio`: Specifies the ratio for splitting the sub-dataset into training and validation sets, default is `0.01`. This parameter is ignored if the sub-dataset has already been split into training and validation sets. For more information, refer to the section on `dataset_split_seed`.
- `--train_dataset_sample`: Samples from the complete training dataset, default is `20000`, to speed up training. This parameter is used to avoid the issue of training time being too long for a single epoch when the dataset is large. LoRA convergence is usually fast and does not require a large number of data samples for fine-tuning. If you specify `-1`, the full training dataset will be used for training, which is typically used in the setting of full-parameter fine-tuning.
- `--system`: The system used in the dialogue template, default is `'you are a helpful assistant!'`.
- `--max_length`: Maximum token length, default is `2048`. This helps to avoid out-of-memory (OOM) issues caused by individual samples that are too long. If a data sample exceeds the `max_length`, the frontmost tokens will be truncated: `input_ids[-max_length:]`.
- `--quantization_bit`: Specifies whether to perform quantization and the number of quantization bits, default is `0`, which means no quantization. Quantization is only supported for the lora fine-tuning method and not for full-parameter fine-tuning.
- `--bnb_4bit_comp_dtype`: When performing 4-bit quantization, we need to dequantize it during the model's forward and backward passes. This parameter specifies the torch_dtype after dequantization. Default is `None`, which means it remains consistent with `dtype`. The possible values are: 'fp16', 'bf16', 'fp32'. This parameter is ignored when `quantization_bit` is 0.
- `--bnb_4bit_quant_type`: The quantization type for 4-bit quantization, default is `'nf4'`. The possible values are: 'nf4', 'fp4'. This parameter is ignored when `quantization_bit` is 0.
- `--bnb_4bit_use_double_quant`: Whether to enable double quantization during 4-bit quantization, default is `True`. This parameter is ignored when `quantization_bit` is 0.
- `--lora_target_modules`: Specifies the LoRA modules, default is `None`, which means it searches for the `lora_TM` (default is qkv) in `MODEL_MAPPING` based on `model_type`. If `ALL` is passed, all linear layers (excluding the head) will be designated as LoRA modules. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--lora_rank`: Default is `8`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--lora_alpha`: Default is `32`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--lora_dropout_p`: Default is `0.0`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--gradient_checkpointing`: Whether to enable gradient checkpointing, default is `False`. This parameter can be used to save GPU memory, although it slightly slows down the training speed. This parameter is particularly effective when `max_length` and `batch_size` are large.
- `--deepspeed_config_path`: Used to specify the path to the DeepSpeed configuration file. Default is `None`, which means DeepSpeed is not enabled. DeepSpeed can help save GPU memory. We have provided a default configuration file for ZeRO-2: `ds_config/zero2.json`.
- `--batch_size`: Batch size during training, default is `1`. Increasing the batch size can improve GPU utilization but may not necessarily speed up training because within a batch, padding is applied to shorter sentences based on the length of the longest sentence in the batch, introducing unnecessary computations.
- `--eval_batch_size`: Batch size during evaluation, default is `None`. If `predict_with_generate` is set to `True`, it is set to `1`; if `predict_with_generate` is `False`, it is set to `batch_size`.
- `--num_train_epochs`: Number of training epochs, default is `1`. If `max_steps >= 0`, it overrides `num_train_epochs`.
- `--max_steps`: Maximum number of training steps, default is `-1`. If `max_steps >= 0`, it overrides `num_train_epochs`.
- `--optim`: Default is `'adamw_torch'`.
- `--learning_rate`: Default is `None`. If `sft_type` is `'lora'`, it is set to `1e-4`; if `sft_type` is `'full'`, it is set to `2e-5`.
- `--weight_decay`: Default is `0.01`.
- `--gradient_accumulation_steps`: Gradient accumulation, default is `16`. `total_batch_size = batch_size * gradient_accumulation_steps * world_size`.
- `--max_grad_norm`: Gradient clipping, default is `1`.
- `--predict_with_generate`: Whether to use a generative approach during evaluation, default is `False`. If set to `False`, it uses `loss` for evaluation. If set to `True`, it uses metrics such as `ROUGE-L` for evaluation. Note that using the generative approach for evaluation is time-consuming, so use it with caution.
- `--lr_scheduler_type`: Default is `'cosine'`.
- `--warmup_ratio`: Ratio of warmup steps to the total training steps, default is `0.05`.
- `--eval_steps`: Perform evaluation every specified number of steps, default is `50`.
- `--save_steps`: Save the model every specified number of steps, default is `None`, which sets it to `eval_steps`.
- `--only_save_model`: Whether to only save the model parameters without storing the intermediate states required for resuming training. The default value is `None`. If `sft_type` is 'lora' and DeepSpeed is not used (deepspeed_config_path is None), it is set to False; otherwise, it is set to True (e.g., when using full parameter fine-tuning or DeepSpeed).
- `--save_total_limit`: The number of checkpoints to save. The default value is `2`, which saves the best and last checkpoints. If set to -1, it saves all checkpoints.
- `--logging_steps`: Number of training steps to print training information (e.g., loss, learning_rate, etc.). Default is `5`.
- `--dataloader_num_workers`: The number of worker processes to use for data loading. The default value is `1`.
- `--push_to_hub`: Whether to synchronize the training checkpoints to the ModelScope Hub. The default value is `False`.
- `--hub_model_id`: The model id of the ModelScope Hub to push to. The default value is `None`, which is set to `f'{model_type}-{sft_type}'`. You can set it to a specific model id or repository name. The user name will be inferred from the `hub_token`. If the remote repository does not exist, a new repository will be created. If it exists, the previous repository will be reused. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_private_repo`: Whether to set the permission of the model repository in the ModelScope Hub to private. The default value is `True`. This parameter only takes effect when `push_to_hub` is set to True.
- `--push_hub_strategy`: The push strategy. The default value is `'push_best'`. Available options are: 'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'. 'push_best' means that the best model will be pushed and overwrite the previous weights every time the weights are saved. 'push_last' means that the last weights will be pushed and overwrite the previous weights every time the weights are saved. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_token`: The SDK token required for pushing to the ModelScope Hub. You can obtain it from https://modelscope.cn/my/myaccesstoken. The default value is `None`, which retrieves the token from the environment variable `MODELSCOPE_API_TOKEN`. This parameter only takes effect when `push_to_hub` is set to True.
- `--test_oom_error`: Used to check if training will encounter an out-of-memory (OOM) error. The default value is `False`. If set to True, the training set will be sorted in reverse order of `max_length` to facilitate OOM testing. This parameter is generally used for testing, so please use it with caution.
- `--use_flash_attn`: Whether to use flash attention. The default value is `None`, which is set to 'auto'. This parameter only takes effect when `model_type.startswith('qwen')` is True. For installation steps of flash attention, please refer to https://github.com/Dao-AILab/flash-attention.
- `--max_new_tokens`: The maximum number of new tokens to generate. The default value is `1024`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--do_sample`: Whether to use sampling during generation. The default value is `True`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--temperature`: The temperature value for sampling during generation. The default value is `0.9`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--top_k`: The value of k for top-k sampling during generation. The default value is `20`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--top_p`: The cumulative probability threshold for top-p sampling during generation. The default value is `0.9`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--repetition_penalty`: The repetition penalty applied during generation. The default value is `1.0`. This parameter only takes effect when `predict_with_generate` is set to True.


### infer.sh Command Line Arguments
- `--model_type`: Default value is `'qwen-7b-chat'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--sft_type`: Default value is `'lora'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--template_type`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--ckpt_dir`: Required field, value is the checkpoint path saved during the SFT phase, e.g., `'/path/to/your/vx_xxx/checkpoint-xxx'`.
- `--eval_human`: Whether to evaluate using the validation set from the dataset or manually evaluate the model. Default value is `False`. This allows us to get an intuitive understanding of the model's performance after fine-tuning.
- `--seed`: Default value is `42`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--dtype`: Default value is `'bf16'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--ignore_args_error`: Default value is `False`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--dataset`: Default value is `'advertise-gen-zh'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--dataset_split_seed`: Default value is `42`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--dataset_test_ratio`: Default value is `0.01`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--show_dataset_sample`: Indicates the number of samples from the validation set to evaluate and display. Default value is `20`. This parameter only takes effect when `eval_human` is set to False.
- `--system`: Default value is `'you are a helpful assistant!'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--max_length`: Default value is `2048`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--quantization_bit`: Default value is 0. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--bnb_4bit_comp_dtype`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--bnb_4bit_quant_type`: Default value is `'nf4'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--bnb_4bit_use_double_quant`: Default value is `True`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--max_new_tokens`: Maximum number of new tokens to generate. Default value is `1024`.
- `--do_sample`: Whether to use greedy decoding or sampling for generation. Default value is `True`.
- `--temperature`: Default value is `0.9`. This parameter only takes effect when `do_sample` is set to True.
- `--top_k`: Default value is `20`. This parameter only takes effect when `do_sample` is set to True.
- `--top_p`: Default value is `0.9`. This parameter only takes effect when `do_sample` is set to True.
- `--repetition_penalty`: Default value is `1.0`.
- `--use_flash_attn`: Default value is `None`, which means 'auto'. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--use_streamer`: Whether to use streaming output. Default value is `True`.
- `--merge_lora_and_save`: Whether to merge the lora weights into the base model and save the complete weights. Default value is `False`. The weights will be saved in a directory named `checkpoint-xxx-merged` at the same level as `ckpt_dir`, e.g., `'/path/to/your/vx_xxx/checkpoint-xxx-merged'`.
- `--save_generation_config`: Whether to save the generation_config used for evaluation as a `generation_config.json` file. Default value is `True`.
