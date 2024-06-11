# LLM Evaluation Documentation

SWIFT supports the eval (evaluation) capability to provide standardized evaluation metrics for the original model and the fine-tuned model.

## Table of Contents

- [Introduction](#Introduction)
- [Environment Setup](#Environment-setup)
- [Evaluation](#Evaluation)
- [Custom Evaluation Set](#Custom-Evaluation-Set)

## Introduction

SWIFT's eval capability utilizes the [EvalScope evaluation framework](https://github.com/modelscope/eval-scope) from the ModelScope community and provides advanced encapsulation to support evaluation needs for various models. Currently, we support the evaluation process for **standard evaluation sets** and **user-defined evaluation sets**. The **standard evaluation sets** include:

- MMLU

> MMLU (Massive Multitask Language Understanding) aims to measure the knowledge gained during pretraining by specifically evaluating models in zero-shot and few-shot settings. This makes the benchmark more challenging and more similar to how we evaluate humans. The benchmark covers 57 subjects across STEM, humanities, and social sciences. Its difficulty ranges from elementary to advanced professional levels, testing world knowledge and problem-solving abilities. The subject range spans traditional fields such as mathematics and history to more specialized domains like law and ethics. The granularity and breadth of topics make the benchmark an ideal choice for identifying model blindspots.
>
> MMLU is an **English evaluation dataset** containing **57 multiple-choice question-answering tasks** [Diversity Benchmark], covering elementary mathematics, American history, computer science, law, etc., covering human knowledge from high school level to expert level. It is currently the mainstream LLM evaluation dataset.

- CEVAL

> C-EVAL is the first comprehensive Chinese evaluation suite, aiming to evaluate the advanced knowledge and reasoning abilities of foundation models in the Chinese context. C-EVAL includes multiple-choice questions at four difficulty levels: middle school, high school, university, and professional. The questions cover 52 different subject areas, ranging from humanities to science and engineering subjects. C-EVAL also comes with C-EVAL HARD, which is a particularly challenging subset of topics from C-EVAL that requires advanced reasoning abilities to solve.

- GSM8K

> GSM8K (Grade School Math 8K) is a dataset containing 8.5K high-quality linguistically diverse elementary school math word problems. The dataset was created to support the task of question-answering on multi-step reasoning problems in elementary mathematics.
>
> GSM8K is a high-quality English elementary math problem test set, containing 7.5K training data and 1K test data. These problems typically require 2-8 steps to solve, effectively evaluating mathematical and logical abilities.

- ARC

> The AI2 Reasoning Challeng(**arc**) dataset is a multiple-choice question-answering dataset containing questions from 3rd to 9th-grade science exams. The dataset is split into two partitions: Easy and Challenge, with the latter containing harder questions requiring reasoning. Most questions have 4 answer choices, with <1% of questions having 3 or 5 answer choices. ARC includes a supporting corpus of 14.3 million KB of unstructured text passages.

- BBH

> BBH (BIG-Bench Hard) is a dataset composed of 23 challenging tasks selected from the BIG-Bench evaluation suite.
>
> BIG-Bench is a diverse test suite aimed at evaluating language model capabilities, including tasks considered to be beyond the current abilities of language models. In the initial BIG-Bench paper, researchers found that the most advanced language models at the time could only outperform the average human rater on 65% of the tasks with a few example prompts.
>
> Therefore, the researchers filtered out the 23 particularly challenging tasks from BIG-Bench where language models failed to surpass human performance, constructing the BBH dataset. These 23 tasks are considered representative challenges that language models still struggle with. Researchers evaluated the effect of thought-chain prompts on improving language model performance on BBH.
>
> Overall, the BBH dataset contains the 23 most challenging tasks from BIG-Bench, aiming to test the limits of language models' capabilities on complex multi-step reasoning problems. Through experiments on BBH, researchers can uncover the benefits of prompting strategies like thought-chains in enhancing language model performance.

## Environment Setup

```shell
pip install ms-swift[eval] -U
```

or install from source code:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[eval]'
```

## Evaluation

Evaluation supports the use of vLLM for acceleration. Here we demonstrate the evaluation of the original model and the LoRA fine-tuned qwen2-7b-instruct.

```shell
# Original model (approximately half an hour on a single A100)
CUDA_VISIBLE_DEVCIES=0 swift eval --model_type qwen2-7b-instruct \
    --eval_dataset ceval mmlu arc gsm8k --infer_backend vllm

# After LoRA fine-tuning
CUDA_VISIBLE_DEVICES=0 swift eval --ckpt_dir qwen2-7b-instruct/vx-xxx/checkpoint-xxx \
    --eval_dataset ceval mmlu arc gsm8k --infer_backend vllm \
    --merge_lora true \
```

You can refer to [here](./Command-line-parameters.md#eval-parameters) for the list of evaluation parameters.

### Evaluation using the deployed method

```shell
# Start deployment using the OpenAI API method
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct

# Evaluate using the API
swift eval --eval_url http://127.0.0.1:8000/v1 --eval_is_chat_model true \
    --model_type qwen2-7b-instruct --eval_dataset ceval mmlu arc gsm8k

# The same applies to the model after LoRA fine-tuning.
```

## Custom Evaluation Set

In addition, we support users to define their own evaluation sets for evaluation. The custom evaluation set must be consistent with the data format (pattern) of an official evaluation set. Below, we will explain step by step how to use your own evaluation set for evaluation.

### Prepare Your Own Evaluation Set

Currently, we support two patterns of evaluation sets: multiple-choice format of CEval and question-answering format of General-QA.

#### Multiple-choice: CEval Format

The CEval format is suitable for scenarios where users have multiple-choice questions. That is, select one correct answer from four options, and the evaluation metric is `accuracy`. It is recommended to **directly modify** the [CEval scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_ceval). This directory contains two files:

```text
default_dev.csv # Used for few-shot evaluation, at least eval_few_shot number of data is required, i.e., this csv can be empty for 0-shot evaluation
default_val.csv # Data used for actual evaluation
```

The CEval csv file needs to be in the following format:

```csv
id,question,A,B,C,D,answer,explanation
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C,1. 目前已知构成动物蛋白质的的氨基酸有20种。
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C,"代谢终产物是指在生物体内代谢过程中产生的无法再被利用的物质，需要通过排泄等方式从体内排出。丙酮酸是糖类代谢的产物，可以被进一步代谢为能量或者合成其他物质，并非代谢终产物。"
```

Here, id is the evaluation sequence number, question is the question, ABCD are the options (leave blank if there are fewer than four options), answer is the correct option, and explanation is the explanation.

The `default` filename is the subset name of the CEval evaluation, which can be changed and will be used in the configuration below.

#### Question-Answering: General-QA

General-QA is suitable for scenarios where users have question-answering tasks, and the evaluation metrics are `rouge` and `bleu`. It is recommended to **directly modify** the [General-QA scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_general_qa). This directory contains
