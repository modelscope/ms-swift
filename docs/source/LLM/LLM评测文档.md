# LLM评测文档

SWIFT支持了eval（评测）能力，用于对原始模型和训练后的模型给出标准化的评测指标。

## 目录

- [能力介绍](#能力介绍)
- [环境准备](#环境准备)
- [评测](#评测)
- [自定义评测集](#自定义评测集)

## 能力介绍

SWIFT的eval能力使用了魔搭社区[评测框架EvalScope](https://github.com/modelscope/eval-scope)，并进行了高级封装以支持各类模型的评测需求。目前我们支持了**标准评测集**的评测流程，以及**用户自定义**评测集的评测流程。其中**标准评测集**包含：

- MMLU

> MMLU（大规模多任务语言理解）旨在通过在zero-shot和few-shot设置中专门评估模型来衡量在预训练期间获得的知识。这使得基准更具挑战性，更类似于我们评估人类的方式。该基准涵盖 STEM、人文科学、社会科学等 57 个科目。它的难度从初级到高级专业水平不等，它考验着世界知识和解决问题的能力。科目范围从传统领域，如数学和历史，到更专业的领域，如法律和道德。主题的粒度和广度使基准测试成为识别模型盲点的理想选择。
>
> MMLU是一个包含 **57个多选问答任务的英文评测** 数据集【多样性基准】，涵盖了初等数学、美国历史、计算机科学、法律等，难度覆盖高中水平到专家水平的人类知识，是目前主流的LLM评测数据集。

- CEVAL

> C-EVAL是第一个全面的中文评估套件，旨在评估基础模型在中文语境下的先进的知识和推理能力。C-EVAL包括四个难度级别的多项选择题：初中、高中、大学和专业。问题涉及涵盖52个不同的学科领域，从人文学科到科学和工程学科不等。C-EVAL还附带有C-EVAL HARD，这是C-EVAL中非常具有挑战性的一部分主题（子集），需要高级推理能力才能解决。

- GSM8K

> GSM8K（小学数学 8K）是一个包含 8.5K 高质量语言多样化小学数学单词问题的数据集。创建该数据集是为了支持对需要多步骤推理的基本数学问题进行问答的任务。
>
> GSM8K是一个高质量的英文小学数学问题测试集，包含 7.5K 训练数据和 1K 测试数据。这些问题通常需要 2-8 步才能解决，有效评估了数学与逻辑能力。

- ARC

> AI2的Reasoning Challeng(**arc**)数据集是一个多项选择问答数据集，包含了从3年级到9年级科学考试中的问题。数据集分为两个分区:Easy和Challenge，后一个分区包含需要推理的更难的问题。大多数问题有4个答案选项，只有<1%的问题有3个或5个答案选项。ARC包含一个支持14.3百万KB的非结构化文本段落。

- BBH

> BBH(BIG-Bench Hard)是从BIG-Bench评估套件中精选出的23个具有挑战性的任务组成的数据集。
>
> BIG-Bench是一个旨在评估语言模型能力的多样化测试集,包含了被认为超出当前语言模型能力范围的各种任务。在最初的BIG-Bench论文中,研究人员发现当时最先进的语言模型只有在65%的任务上能够通过少量示例提示的方式超过平均人类评估员的表现。
>
> 因此,研究人员从BIG-Bench中筛选出那23个语言模型未能胜过人类的特别棘手的任务,构建了BBH数据集。这23个任务被认为是语言模型仍面临挑战的代表性难题。研究人员在BBH上评估了思维链提示对提升语言模型表现的效果。
>
> 总的来说,BBH数据集包含了BIG-Bench中最具挑战性的23项任务,旨在检验语言模型在复杂多步推理问题上的能力极限。通过在BBH上的实验,研究人员能够发现思维链等提示策略对提高语言模型性能的助益。

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

评测的命令非常简单，只需要使用如下命令即可：

```shell
# 使用arc评测，每个子数据集限制评测10条，推理backend使用pt
swift eval \
    --model_type "qwen-7b-chat" \
    --eval_dataset arc \
    --eval_limit 10 \
    --infer_backend pt
```

评测的参数列表可以参考[这里](./命令行参数.md#eval参数)。

评测结果展示如下：

```text
2024-04-10 17:18:45,861 - llmuses - INFO - *** Report table ***
+---------+-----------+
| Model   | arc       |
+=========+===========+
|         | 0.8 (acc) |
+---------+-----------+
Final report:{'report': [{'name': 'arc', 'metric': 'WeightedAverageAccuracy', 'score': 0.8, 'category': [{'name': 'DEFAULT', 'score': 0.8, 'subset': [{'name': 'ARC-Challenge', 'score': 0.8}]}], 'total_num': 10}], 'generation_info': {'time': 80.44219398498535, 'tokens': 743}}
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
        "dataset": "eval_example/custom_general_qa", # 该评测集的目录
        "subset_list": ["default"] # 需要评测的子数据集，即上面的`default_x`文件名
    },
    {
        "name": "custom_ceval",
        "pattern": "ceval",
        "dataset": "eval_example/custom_ceval",
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
