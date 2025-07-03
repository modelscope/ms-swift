# Evaluation

SWIFT supports eval (evaluation) capabilities to provide standardized evaluation metrics for both raw models and trained models.

## Capability Introduction

SWIFT's eval capability utilizes the EvalScope evaluation framework from the Magic Tower community, which has been advanced in its encapsulation to support the evaluation needs of various models.

> Note: EvalScope supports many other complex capabilities, such as [model performance evaluation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html), so please use the EvalScope framework directly.

Currently, we support the evaluation process of **standard evaluation datasets** as well as the evaluation process of **user-defined** evaluation datasets. The **standard evaluation datasets** are supported by three evaluation backends:

Below are the names of the supported datasets. For detailed information on the datasets, please refer to [all supported datasets](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset.html).

1. Native (default):

    Primarily supports pure text evaluation, while **supporting** visualization of evaluation results.
    ```text
    'arc', 'bbh', 'ceval', 'cmmlu', 'competition_math',
    'general_qa', 'gpqa', 'gsm8k', 'hellaswag', 'humaneval',
    'ifeval', 'iquiz', 'mmlu', 'mmlu_pro',
    'race', 'trivia_qa', 'truthful_qa'
    ```

2. OpenCompass:

    Primarily supports pure text evaluation, currently **does not support** visualization of evaluation results.
    ```text
    'obqa', 'cmb', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
    'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
    'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
    'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
    'ARC_e', 'COPA', 'ARC_c', 'DRCD'
    ```

3. VLMEvalKit:

    Primarily supports multimodal evaluation and currently **does not support** visualization of evaluation results.
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
    'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video', 'Video-MME'
    ```

## Environment Preparation

```shell
pip install ms-swift[eval] -U
```

Or install from source:

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[eval]'
```

## Evaluation

Supports four methods of evaluation: pure text evaluation, multimodal evaluation, URL evaluation, and custom dataset evaluation.

**Basic Example**

```shell
CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eval_backend Native \
    --infer_backend pt \
    --eval_limit 10 \
    --eval_dataset gsm8k
```
Where:
- model: Can specify a local model path or a model ID on modelscope
- eval_backend: Options are Native, OpenCompass, VLMEvalKit; default is Native
- infer_backend: Options are pt, vllm, sglang, lmdeploy; default is pt
- eval_limit: Sample size for each evaluation set; default is None, which means using all data; can be used for quick validation
- eval_dataset: Evaluation dataset(s); multiple datasets can be set, separated by spaces

**Complex Evaluation Example**

```shell
CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eval_backend Native \
    --infer_backend pt \
    --eval_limit 10 \
    --eval_dataset gsm8k \
    --dataset_args '{"gsm8k": {"few_shot_num": 0, "filters": {"remove_until": "</think>"}}}' \
    --eval_generation_config '{"max_tokens": 512, "temperature": 0}' \
    --extra_eval_args '{"ignore_errors": true, "debug": true}'
```

For a specific list of evaluation parameters, please refer to [here](./Command-line-parameters.md#evaluation-arguments).

## Evaluation During Training

SWIFT supports using EvalScope to evaluate the current model during the training process, allowing for timely understanding of the model's training effectiveness.

**Basic Example**

```shell
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --train_type "lora" \
  --dataset "AI-ModelScope/alpaca-gpt4-data-zh#100" \
  --torch_dtype "bfloat16" \
  --num_train_epochs "1" \
  --per_device_train_batch_size "1" \
  --learning_rate "1e-4" \
  --lora_rank "8" \
  --lora_alpha "32" \
  --target_modules "all-linear" \
  --gradient_accumulation_steps "16" \
  --save_steps "50" \
  --save_total_limit "5" \
  --logging_steps "5" \
  --max_length "2048" \
  --eval_strategy "steps" \
  --eval_steps "5" \
  --per_device_eval_batch_size "5" \
  --eval_use_evalscope \
  --eval_dataset "gsm8k" \
  --eval_dataset_args '{"gsm8k": {"few_shot_num": 0}}' \
  --eval_limit "10"
```

Note that the launch command is `sft`, and the evaluation-related parameters include:
- eval_strategy: Evaluation strategy. Defaults to None, following the `save_strategy` policy
- eval_steps: Defaults to None. If an evaluation dataset exists, it follows the `save_steps` policy
- eval_use_evalscope: Whether to use evalscope for evaluation, this parameter needs to be set to enable evaluation
- eval_dataset: Evaluation datasets, multiple datasets can be set, separated by spaces
- eval_dataset_args: Evaluation dataset parameters in JSON format, parameters for multiple datasets can be set
- eval_limit: Number of samples from the evaluation dataset
- eval_generation_config: Model inference configuration during evaluation, in JSON format, default is `{'max_tokens': 512}`

More evaluation examples can be found in [examples](https://github.com/modelscope/ms-swift/tree/main/examples/eval).

## Custom Evaluation Datasets

This framework supports two predefined dataset formats: multiple-choice questions (MCQ) and question-and-answer (QA). The usage process is as follows:

*Note: When using a custom evaluation, the `eval_backend` parameter must be set to `Native`.*

### Multiple-Choice Question Format (MCQ)
This format is suitable for scenarios involving multiple-choice questions, and the evaluation metric is accuracy.

**Data Preparation**

Prepare a CSV file in the multiple-choice question format, structured as follows:

```text
mcq/
├── example_dev.csv  # (Optional) The filename should follow the format `{subset_name}_dev.csv` for few-shot evaluation
└── example_val.csv  # The filename should follow the format `{subset_name}_val.csv` for the actual evaluation data
```

The CSV file should follow this format:

```text
id,question,A,B,C,D,answer
1,Generally speaking, the amino acids that make up animal proteins are____,4 types,22 types,20 types,19 types,C
2,Among the substances present in the blood, which is not a metabolic end product?____,Urea,Uric acid,Pyruvate,Carbon dioxide,C
```

Where:
- `id` is an optional index
- `question` is the question
- `A`, `B`, `C`, `D`, etc. are the options, with a maximum of 10 options
- `answer` is the correct option

**Launching Evaluation**

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eval_backend Native \
    --infer_backend pt \
    --eval_dataset general_mcq \
    --dataset_args '{"general_mcq": {"local_path": "/path/to/mcq", "subset_list": ["example"]}}'
```

Where:
- `eval_dataset` should be set to `general_mcq`
- `dataset_args` should be set with:
    - `local_path` as the path to the custom dataset folder
    - `subset_list` as the name of the evaluation dataset, taken from the `*_dev.csv` mentioned above

**Running Results**

```text
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_mcq | AverageAccuracy | example  |    12 |  0.5833 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
```

## Question-and-Answer Format (QA)
This format is suitable for scenarios involving question-and-answer, and the evaluation metrics are `ROUGE` and `BLEU`.

**Data Preparation**

Prepare a JSON Lines file in the question-and-answer format, containing one file in the following structure:

```text
qa/
└── example.jsonl
```

The JSON Lines file should follow this format:

```json
{"query": "What is the capital of China?", "response": "The capital of China is Beijing"}
{"query": "What is the highest mountain in the world?", "response": "It is Mount Everest"}
{"query": "Why can't penguins be seen in the Arctic?", "response": "Because most penguins live in Antarctica"}
```

**Launching Evaluation**

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 \
swift eval \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eval_backend Native \
    --infer_backend pt \
    --eval_dataset general_qa \
    --dataset_args '{"general_qa": {"local_path": "/path/to/qa", "subset_list": ["example"]}}'
```

Where:
- `eval_dataset` should be set to `general_qa`
- `dataset_args` is a JSON string that needs to be set with:
    - `local_path` as the path to the custom dataset folder
    - `subset_list` as the name of the evaluation dataset, taken from the `*.jsonl` mentioned above

**Running Results**

```text
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Model               | Dataset     | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+=============+=================+==========+=======+=========+=========+
| Qwen2-0.5B-Instruct | general_qa  | bleu-1          | default  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-2          | default  |    12 |  0.1451 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-3          | default  |    12 |  0.0625 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | bleu-4          | default  |    12 |  0.0556 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-f       | default  |    12 |  0.3441 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-p       | default  |    12 |  0.2393 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-1-r       | default  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-f       | default  |    12 |  0.2062 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-p       | default  |    12 |  0.1453 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-2-r       | default  |    12 |  0.6167 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-f       | default  |    12 |  0.333  | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-p       | default  |    12 |  0.2324 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
| Qwen2-0.5B-Instruct | general_qa  | rouge-l-r       | default  |    12 |  0.8889 | default |
+---------------------+-------------+-----------------+----------+-------+---------+---------+
```
