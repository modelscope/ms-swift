# LLM评测文档

SWIFT支持了eval（评测）能力，用于对原始模型和训练后的模型给出标准化的评测指标。

## 目录

- [能力介绍](#能力介绍)
- [环境准备](#环境准备)
- [评测](#评测)
- [自定义评测集](#自定义评测集)

## 能力介绍

SWIFT的eval能力使用了魔搭社区[评测框架EvalScope](https://github.com/modelscope/eval-scope)，以及[Open-Compass](https://hub.opencompass.org.cn/home)，并进行了高级封装以支持各类模型的评测需求。目前我们支持了**标准评测集**的评测流程，以及**用户自定义**评测集的评测流程。其中**标准评测集**包含：

纯文本评测：
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
数据集的具体介绍可以查看：https://hub.opencompass.org.cn/home

多模态评测：
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
数据集的具体介绍可以查看：https://github.com/open-compass/VLMEvalKit


> 首次评测时会自动下载数据集文件：https://www.modelscope.cn/datasets/swift/evalscope_resource/files
> 如果下载失败可以手动下载放置本地路径, 具体可以查看eval的日志输出.

## 环境准备

```shell
pip install ms-swift[eval] -U
```

或从源代码安装：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[eval]'
```

## 评测

评测支持使用vLLM加速. 这里展示对原始模型和LoRA微调后的qwen2-7b-instruct进行评测.

```shell
# 原始模型 (单卡A100大约需要半小时)
CUDA_VISIBLE_DEVCIES=0 swift eval --model_type qwen2-7b-instruct \
    --eval_dataset ARC_e --infer_backend vllm

# LoRA微调后
CUDA_VISIBLE_DEVICES=0 swift eval --ckpt_dir qwen2-7b-instruct/vx-xxx/checkpoint-xxx \
    --eval_dataset ARC_e --infer_backend vllm \
    --merge_lora true \
```

评测的参数列表可以参考[这里](./命令行参数.md#eval参数)。

请注意：评测结果会存储在{--eval_output_dir}/{--name}/{时间戳}下, 如果用户没有改变存储配置，则默认路径在:
```text
当前目录(`pwd`路径)/eval_outputs/default/20240628_190000/xxx
```


### 使用部署的方式评测

```shell
# 使用OpenAI API方式启动部署
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct

# 使用API进行评测
# 如果是非swift部署, 则需要额外传入`--eval_is_chat_model true --model_type qwen2-7b-instruct`
swift eval --eval_url http://127.0.0.1:8000/v1 --eval_dataset ARC_e

# LoRA微调后的模型同理
```

## 自定义评测集

除此之外，我们支持了用户自定义自己的评测集。自定义评测集必须和某个官方评测集数据格式（pattern）保持一致。下面我们按步骤讲解如何使用自己的评测集进行评测。

### 写好自己的评测集

目前我们支持两种pattern的评测集：选择题格式的CEval和问答题格式的General-QA

#### 选择题：CEval格式

CEval格式适合用户是选择题的场景。即从四个选项中选择一个正确的答案，评测指标是`accuracy`。建议**直接修改**[CEval脚手架目录](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_ceval)。该目录包含了两个文件：

```text
default_dev.csv # 用于fewshot评测，至少要具有入参的eval_few_shot条数据，即如果是0-shot评测，该csv可以为空
default_val.csv # 用于实际评测的数据
```

CEval的csv文件需要为下面的格式：

```csv
id,question,A,B,C,D,answer,explanation
1,通常来说，组成动物蛋白质的氨基酸有____,4种,22种,20种,19种,C,1. 目前已知构成动物蛋白质的的氨基酸有20种。
2,血液内存在的下列物质中，不属于代谢终产物的是____。,尿素,尿酸,丙酮酸,二氧化碳,C,"代谢终产物是指在生物体内代谢过程中产生的无法再被利用的物质，需要通过排泄等方式从体内排出。丙酮酸是糖类代谢的产物，可以被进一步代谢为能量或者合成其他物质，并非代谢终产物。"
```

其中，id是评测序号，question是问题，ABCD是可选项（如果选项少于四个则对应留空），answer是正确选项，explanation是解释。

其中的`default`文件名是CEval评测的子数据集名称，可更换，下面的配置中会用到。

#### 问答题：General-QA

General-QA适合用户是问答题的场景，评测指标是`rouge`和`bleu`。建议**直接修改**[General-QA脚手架目录](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_general_qa)。该目录包含了一个文件：

```text
default.jsonl
```

该jsonline文件需要为下面的格式：

```jsonline
{"history": [], "query": "中国的首都是哪里？", "response": "中国的首都是北京"}
{"history": [], "query": "世界上最高的山是哪座山？", "response": "是珠穆朗玛峰"}
{"history": [], "query": "为什么北极见不到企鹅？", "response": "因为企鹅大多生活在南极"}
```

注意`history`目前为保留字段，尚不支持。

### 定义一个配置文件传入eval命令

定义好上面的文件后，需要写一个json文件传入eval命令中。建议直接修改[官方配置脚手架文件](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_config.json)。该文件内容如下：

```json
[
    {
        "name": "custom_general_qa", # 评测项名称，可以随意指定
        "pattern": "general_qa", # 该评测集的pattern
        "dataset": "eval_example/custom_general_qa", # 该评测集的目录，强烈建议使用绝对路径防止读取失败
        "subset_list": ["default"] # 需要评测的子数据集，即上面的`default_x`文件名
    },
    {
        "name": "custom_ceval",
        "pattern": "ceval",
        "dataset": "eval_example/custom_ceval", # 该评测集的目录，强烈建议使用绝对路径防止读取失败
        "subset_list": ["default"]
    }
]
```

下面就可以传入这个配置文件进行评测了：

```shell
# 使用arc评测，每个子数据集限制评测10条，推理backend使用pt
# cd examples/pytorch/llm
# eval_dataset也可以设置值，官方数据集和自定义数据集一起跑
swift eval \
    --model_type "qwen-7b-chat" \
    --eval_dataset no \
    --infer_backend pt \
    --custom_eval_config eval_example/custom_config.json
```

运行结果如下：

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
