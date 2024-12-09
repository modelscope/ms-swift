# Evaluation

SWIFT supports eval(evaluation) capabilities to provide standardized assessment metrics for both the raw model and the trained model.

## Capability Introduction

SWIFT's eval capability uses the [evalution framework EvalScope](https://github.com/modelscope/eval-scope) from the ModelScope, with high-level encapsulation to meet various model evaluation needs.

Currently, we support evaluation processes for **standard evaluation sets** as well as **user-defined** evaluation sets. The **standard evaluation sets** include:

> Note: EvalScope supports many other complex capabilities, such as model performance evaluation. Please use the EvalScope framework directly for those features.

Pure Text Evaluation:
```text
'obqa', 'cmb', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
For detailed information on the datasets, please visit: https://hub.opencompass.org.cn/home

Multimodal Evaluation:
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN',
'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11',
'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2',
'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL',
'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar',
'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL',
'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI',
'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST',
'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500',
'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL',
'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video', 'Video-MME',
'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN',
'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11',
'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST',
'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL',
'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```
For detailed information on the datasets, please visit: https://github.com/open-compass/VLMEvalKit

## Environment Preparation

```shell
pip install ms-swift[eval] -U
```

Or install from the source code:

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[eval]'
```

## Evaluation

We support four types of evaluation: pure text evaluation, multimodal evaluation, URL evaluation, and custom dataset evaluation.

For sample evaluations, please refer to [examples](https://github.com/modelscope/ms-swift/tree/main/examples/eval).

The list of evaluation parameters can be found [here](Commend-line-parameters#评测参数).

## Custom Evaluation Sets

Note: The documentation below is not supported in version 3.0; please use version 2.x for evaluation.

Additionally, we support users to create their own evaluation sets. Custom evaluation sets must match the data format (pattern) of an official evaluation set. Below we explain how to use your evaluation set step by step.

### Create Your Custom Evaluation Set

Currently, we support two patterns for evaluation sets: the multiple-choice format CEval and the question-answer format General-QA.

#### Multiple-Choice: CEval Format

The CEval format is suitable for multiple-choice scenarios, where one correct answer is selected from four options. The evaluation metric is `accuracy`. It is recommended to **directly modify** the [CEval scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_ceval). This directory contains two files:

```text
default_dev.csv # For fewshot evaluation, it must have at least the number of data entries specified by the input parameter `eval_few_shot`. For 0-shot evaluation, this CSV can be empty.
default_val.csv # For actual evaluation data.
```

The CSV file for CEval needs to follow this format:

```csv
id,question,A,B,C,D,answer,explanation
1,Generally speaking, there are ____, 4, 22, 20, 19 kinds of amino acids that make up animal proteins, C,1. Currently, it is known that there are 20 kinds of amino acids that constitute animal proteins.
2. Among the substances present in the blood, the one that is not a metabolic end product is ____. Urea, Uric acid, Pyruvic acid, Carbon dioxide, C. "Metabolic end products refer to substances that are produced during metabolic processes in living organisms and cannot be utilized further, needing to be expelled from the body through excretion or other means. Pyruvic acid is a product of carbohydrate metabolism and can be further metabolized for energy or synthesized into other substances, thus it is not a metabolic end product."
```

Here, `id` is the sequence number, `question` is the question, A, B, C, D are the options (leave empty if there are less than four options), `answer` is the correct option, and `explanation` is the explanation.

The name of `default` files is the name of the CEval sub-dataset, which can be changed and will be used in the configuration below.

#### Question-Answer: General-QA

General-QA is suitable for question-answer scenarios, with evaluation metrics being `rouge` and `bleu`. It is recommended to **directly modify** the [General-QA scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_general_qa). This directory contains one file:

```text
default.jsonl
```

This jsonline file must follow this format:

```jsonline
{"history": [], "query": "What is the capital of China?", "response": "The capital of China is Beijing."}
{"history": [], "query": "Which is the highest mountain in the world?", "response": "It is Mount Everest."}
{"history": [], "query": "Why can't penguins be seen in the Arctic?", "response": "Because penguins mostly live in Antarctica."}
```

Note that `history` is currently a reserved field and is not supported yet.

### Define a Configuration File for the Eval Command

After defining the above files, a JSON file needs to be created to pass to the eval command. It is suggested to directly modify the [official configuration scaffold file](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_config.json). The file content should be as follows:

```json
[
    {
        "name": "custom_general_qa", # Name of the evaluation item, can be freely specified
        "pattern": "general_qa", # Pattern of the evaluation set
        "dataset": "eval_example/custom_general_qa", # Directory of the evaluation set, it is strongly recommended to use an absolute path to prevent read failure
        "subset_list": ["default"] # The sub-dataset to evaluate, i.e., the above `default_x` filename
    },
    {
        "name": "custom_ceval",
        "pattern": "ceval",
        "dataset": "eval_example/custom_ceval", # Directory of the evaluation set, it is strongly recommended to use an absolute path to prevent read failure
        "subset_list": ["default"]
    }
]
```

You can then pass this configuration file for evaluation:

```shell
# Use arc evaluation, limit 10 evaluation entries per sub-dataset, inference backend using pt
# cd examples/pytorch/llm
# eval_dataset can also be set, official datasets and custom datasets can run together
swift eval \
    --model_type "qwen-7b-chat" \
    --eval_dataset no \
    --infer_backend pt \
    --custom_eval_config eval_example/custom_config.json
```

The output will be as follows:

```text
2024-04-10 17:21:33,275 - llmuses - INFO - *** Report table ***
+------------------------------+----------------+---------------------------------+
| Model                        | custom_ceval   | custom_general_qa               |
+==============================+================+=================================+
| qa-custom_ceval_qwen-7b-chat | 1.0 (acc)      | 0.8888888888888888 (rouge-1-r)  |
|                              |                | 0.33607503607503614 (rouge-1-p) |
|                              |                | 0.40616618868713145 (rouge-1-f) |
|                              |                | 0.39999999999999997 (rouge-2-r) |
|                              |                | 0.27261904761904765 (rouge-2-p) |
|                              |                | 0.30722525589718247 (rouge-2-f) |
|                              |                | 0.8333333333333334 (rouge-l-r)  |
|                              |                | 0.30742204655248134 (rouge-l-p) |
|                              |                | 0.3586824745225346 (rouge-l-f)  |
|                              |                | 0.3122529644268775 (bleu-1)     |
|                              |                | 0.27156862745098037 (bleu-2)    |
|                              |                | 0.25 (bleu-3)                   |
|                              |                | 0.2222222222222222 (bleu-4)     |
+------------------------------+----------------+---------------------------------+
Final report:{'report': [{'name': 'custom_general_qa', 'metric': 'WeightedAverageBLEU', 'score': {'rouge-1-r': 0.8888888888888888, 'rouge-1-p': 0.33607503607503614, 'rouge-1-f': 0.40616618868713145, 'rouge-2-r': 0.39999999999999997, 'rouge-2-p': 0.27261904761904765, 'rouge-2-f': 0.30722525589718247, 'rouge-l-r': 0.8333333333333334, 'rouge-l-p': 0.30742204655248134, 'rouge-l-f': 0.3586824745225346, 'bleu-1': 0.3122529644268775, 'bleu-2': 0.27156862745098037, 'bleu-3': 0.25, 'bleu-4': 0.2222222222222222}, 'category': [{'name': 'DEFAULT', 'score': {'rouge-1-r': 0.8888888888888888, 'rouge-1-p': 0.33607503607503614, 'rouge-1-f': 0.40616618868713145, 'rouge-2-r': 0.39999999999999997, 'rouge-2-p': 0.27261904761904765, 'rouge-2-f': 0.30722525589718247, 'rouge-l-r': 0.8333333333333334, 'rouge-l-p': 0.30742204655248134, 'rouge-l-f': 0.3586824745225346, 'bleu-1': 0.3122529644268775, 'bleu-2': 0.27156862745098037, 'bleu-3': 0.25, 'bleu-4': 0.2222222222222222}, 'subset': [{'name': 'default', 'score': {'rouge-1-r': 0.8888888888888888, 'rouge-1-p': 0.33607503607503614, 'rouge-1-f': 0.40616618868713145, 'rouge-2-r': 0.39999999999999997, 'rouge-2-p': 0.27261904761904765, 'rouge-2-f': 0.30722525589718247, 'rouge-l-r': 0.8333333333333334, 'rouge-l-p': 0.30742204655248134, 'rouge-l-f': 0.3586824745225346, 'bleu-1': 0.3122529644268775, 'bleu-2': 0.27156862745098037, 'bleu-3': 0.25, 'bleu-4': 0.2222222222222222}}]}], 'total_num': 3}, {'name': 'custom_ceval', 'metric': 'WeightedAverageAccuracy', 'score': 1.0, 'category': [{'name': 'DEFAULT', 'score': 1.0, 'subset': [{'name': 'default', 'score': 1.0}]}], 'total_num': 2}], 'generation_info': {'time': 34.23462510108948, 'tokens': 219}}
```
