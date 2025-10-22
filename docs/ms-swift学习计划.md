# 注释prompt

- 类注释prompt

```
你需要完成的任务和目标：为Template类生成注释，要求准确详细，风格清晰易懂，能够使得阅读者轻易理解当前类的含义和功能。
对整体类的注释需要包括以下部分：
1、类功能
2、（1）继承关系说明（如果没有继承可以跳过）；（2）应用场景
3、使用示例
对类中各个方法的注释包括以下部分：
1、函数功能
2、参数
3、返回值
4、使用示例
5、如果已经有英文或者中文注释，则该写到以上的相应部分，比如函数功能、参数、返回值和使用示例部分。
6、对方法中的代码体逐行进行注释
```



- 函数注释prompt

```
你需要完成的任务和目标：为load_image方法生成注释，要求准确详细，风格清晰易懂，能够使得阅读者轻易理解当前函数的含义和功能。
对整体函数的注释需要包括以下部分：
1、函数功能
2、参数
3、返回值
4、使用示例
5、如果已经有英文或者中文注释，则该写到以上的相应部分，比如函数功能、参数、返回值和使用示例部分。
6、对方法中的代码体逐行进行注释
```



脚本注释prompt:

```
你需要完成的任务和目标：为template_meta.py脚本生成注释，要求准确详细，风格清晰易懂，能够使得阅读者轻易理解当前类的含义和功能。
对整体类的注释需要包括以下部分：
1、类功能
2、（1）继承关系说明（如果没有继承可以跳过）；（2）应用场景
3、使用示例
对类中各个方法的注释包括以下部分：
1、函数功能
2、参数
3、返回值
4、使用示例
5、如果已经有英文或者中文注释，则该写到以上的相应部分，比如函数功能、参数、返回值和使用示例部分。
6、对方法中的代码体逐行进行注释
```





```
需要你完成的任务和目标是：参考infer_client.py中的注释风格，给template/base.py的每一行代码生成详细、准确的注释，使得阅读代码的人能够轻易理解代码的含义和功能。
重点注意：每一行代码都需要进行注释！每一行代码都需要进行注释！每一行代码都需要进行注释！
```



```
请你继续注释utils.py脚本，我要求的是：每一行代码都需要进行注释！每一行代码都需要进行注释！每一行代码都需要进行注释！
```



```
请重新注释_infer_full方法，因为目前的注释并没有让我理解该方法的作用和具体运行过程。
要求：
1、在函数注释中，如果入参涉及Tensor，则给出相关Tensor的形状；
2、在示例中，给出更加具体、但是简单易懂的实际例子；
3、在代码体中（具体代码逻辑）中，给出更加详细的注释，说明这一行代码在做什么事情；
4、如果当前行涉及到tensor，则注释tensor的shape
```



```
需要你完成的任务和目标是：参考infer_client.py中的注释风格，给template/base.py的每一行代码生成详细、准确的注释，使得阅读代码的人能够轻易理解代码的含义和功能，注意是每一行代码都需要进行注释！
具体的要求是：
1、给出模块功能概述。
2、如果是类，需要给出类功能说明，给出更加具体、但是简单易懂的实际例子；
3、如果是函数，需要给出函数功能说明，给出更加具体、但是简单易懂的实际例子；
4、在代码体中（具体代码逻辑）中，给出更加详细的注释，说明这一行代码在做什么事情；
5、如果当前行涉及到tensor，则注释tensor的shape。
```



```
维护和优化IPC文件解析和BGA/金手指属性识别功能
1、优化识别模型，将在测试集上识别准确率提升到97%以上；
2、优化模型和几何规则的两级判定策略，在兴森实际数据上，提升BGA准确率到98%以上，漏识别率在1%以下；提升BGA准确率到90%以上，漏识别率在3%以下；
3、维护SDK，保证稳定性，不出现运行bug。
```



```
"1、获取图像数据，进行清洗以及数据增广；
2、训练BGA/金手指/负样本的图像分类模型，对准确率进行调优直到满足预期性能；
3、实现从IPC文件中根据器件分组解析出图像的功能模块；
4、实现拉通从IPC文件解析到属性检测的SDK；"


核心思路是，根据客户持续提供的实际数据，发现在实际数据上的错识别和漏识别问题，优化识别模型、几何规则和SDK。
具体的措施是：
1、使用当前版本的识别系统，对客户提供的实际数据进行检测，统计每个订单和每个组件的识别结果，并按类别计算正确率和漏识别率，整理到表格中记录。
2、分析实际数据的检测结果，调试其中错误识别和漏识别的案例，定位到具体的错误原因。
3、针对模型识别错误的部分，解析并清洗客户提供的实际数据，如果需要则进行数据增强，提升模型对于某些边界情况的识别能力。
4、针对几何特征判别错误的部分，根据错误案例调整几何判别策略，在不影响原有判别能力的前提，进一步增强对于边界情况的检测能力。
5、针对极少数由于IPC文件缺少形状信息导致的错误，与客户进行沟通，尝试从后面应用步骤结合其他信息进行判别。
```



```

```





# 目标/规划

## 目标

三阶段：

- 第一阶段：能跑通示例、写基础测试并符合项目贡献规范。

- 第二阶段：熟练使用 swift 进行训练、推理、评测与部署。

- 第三阶段：理解 swift/ 核心代码结构与数据/模型/训练管道，能做二次开发（加模型/数据集/算法）。
- 第四阶段：如臂使指，熟悉swift的每一个细节，能够根据实际需求进行扩展。

## 规划

一、结合cursor，精读并注释全部源代码

二、修改源代码，编译，观察修改效果，加深理解

三、结合swift源代码阅读Qwen系列模型代码

四、处于学习和理解的目的，使用小批量数据，使用swift框架训练Qwen系列模型



# 顺序

对应顶层目录/文件的学习顺序：

1) README_CN.md/README.md（总
2) 
3) 
4) 能力边界，使用中文版）
5) docs/（阅读相关文档：微调、RLHF、量化、vLLM/SGLang/LmDeploy、评测 EvalScope、UI）
6) requirements/ + requirements.txt（依赖与可选组件）
7) Makefile、scripts/、.dev_scripts/、.pre-commit-config*.yaml（本地常用命令与工程规范）
8) examples/（先跑通：训练/推理/评测/部署）
9) setup.py → 定位入口：swift/cli/main.py、swift/cli/_megatron/main.py
10) swift/（源码主线：CLI → 配置解析 → 模型/数据加载 → 训练循环 → 加速/量化/评测/部署）
11) docs/（深挖：微调、RLHF、量化、vLLM/SGLang/LmDeploy、评测 EvalScope、UI）
12) tests/（验证思路与用法，学习最小可复现实例）
13) CONTRIBUTING*.md、CODE_OF_CONDUCT.md、.github/（提交流程与质量保障）
14) setup.cfg、MANIFEST.in、LICENSE、asset/（工程化与发布收尾）



# 可执行计划

时间：7–10 天（2025/8/18 - 2025/8/31）

每步都给出“目标/材料/动作/产出”。建议每天 2–4 小时，按进度可并行推进 4/5/6 步。

## D1：总览与环境

一、目标：清楚项目做什么、支持什么、怎么装。

二、材料：

- README_CN.md 总览文档
- setup.py 打包安装脚本

三、动作：

1、通读“简介/安装/快速开始/...”，记录涉及的关键词（LoRA/QLoRA、DPO/GRPO、vLLM/SGLang、EvalScope、Megatron等）。`DONE`

2、本地安装可运行环境：`DONE`

```shell
# 克隆与开发安装
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# 基础依赖 + 常用可选依赖
pip install -r requirements/framework.txt
pip install -r requirements/eval.txt -r requirements/swanlab.txt

# 代码规范工具（可选）
pre-commit install
```

3、阅读setup.py，结合AI工具进行详细注释。`DONE`

四、产出：

1、能 import swift，了解项目模块全貌

2、详细注释后的steup.py



## D2：docs/source[GetStarted, Instructio]

一、目标：清楚具体怎么使用swift框架。

二、材料：docs/source的[GetStarted, Instructio]部分

三、动作：

1、GetStarted	`DONE`

2、Instruction	`DONE`

四、产出：注释、修改后的[GetStarted, Instructio]部分。



## D3：BestPractices

一、目标：按照最佳实验的教程，依次实操每一个任务。

二、材料：docs/source的[BestPractices]部分 + 代码

三、动作：

1、Qwen3最佳实践	`DONE`

2、快速训练VL模型	`DONE`

3、GRPO多模态训练	`ING-待训练`

4、GRPO完整流程	`ING-待训练`

5、GRPO代码训练	`ING-待训练`

6、Embedding训练	`ING-待训练`

7、Reranker训练	`ING-待训练`

8、更多最佳实践	`ING-待训练`

四、产出：

1、Qwen3最佳实践：阅读完 Qwen3最佳实践，跑通 qwen3.ipynb

2、其它的阅读了文档和代码，但都待训练，留到后面进行。



## D4：跑通核心示例

一、目标：把最小闭环跑起来（训练/推理/评测/部署至少各一个）。

二、材料：examples/

三、动作：

1、训练：train

| submodule             | status |
| --------------------- | ------ |
| tuners                | DONE   |
| qlora                 | DONE   |
| pretrain              | DONE   |
| streaming             | DONE   |
| think_model           | DONE   |
| sequence_parallel     | DONE   |
| seq_cls               | DONE   |
| rlhf                  | DONE   |
| rft                   | DONE   |
| reranker              | DONE   |
| qlora                 | DONE   |
| predict_with_generate | DONE   |
| plugins               | DONE   |
| padding_free          | DONE   |
| packing               | DONE   |
| optimizer             | DONE   |
| new_special_tokens    | DONE   |
| multimodal            | DONE   |
| multi-node            | DONE   |
| multi-gpu             | DONE   |
| moe                   | DONE   |
| megatron              | DONE   |
| liger                 | DONE   |
| grpo                  | DONE   |
| full                  | DONE   |
| flash_attention_3     | DONE   |
| embedding             | DONE   |
| base_to_chat          | DONE   |
| all_to_all            | DONE   |
| agent                 | DONE   |

`DONE`

2、推理：infer

`DONE`

| submodule | status |
| --------- | ------ |
|           |        |

3、导出/量化：export

`DONE`

4、部署：deploy

`DONE`

5、评测：eval

`DONE`

6、models



四、产出：保存日志/权重/评测结果与复现实验命令。





### 第4–5天：从入口到训练主线

- 目标：掌握 CLI → 配置 → 数据/模型 → 训练循环 的主数据流。

- 材料：setup.py（entry_points）、swift/、examples/、docs/

- 动作：

- 从 setup.py 的 entry points 入手，定位：

- swift → swift/cli/main.py（cli_main）

- megatron → swift/cli/_megatron/main.py

- 顺藤摸瓜：从 CLI 的某个子命令（如 sft）跳转至对应实现，梳理以下问题：

- 参数解析与配置合并（命令行/默认配置/环境变量）如何组织？

- 模型加载（ModelScope/HuggingFace/本地）、权重精度与量化策略如何注入？

- 数据集加载/切分/预处理的抽象与接口？

- 训练循环与回调（日志、评估、保存、早停）、轻量化（LoRA/DoRA 等）如何拼装？

- 分布式/并行（DDP/FSDP/DeepSpeed/Megatron）在何处切换？

- 对照 examples/train/* 与 docs/ 中 SFT/DPO/GRPO 篇章，建立“命令 ↔ 代码位置 ↔ 关键类/函数”的映射笔记。

- 产出：一张“训练主线流程图”和关键代码路径清单。

### 第6天：推理/部署/评测/量化模块

- 目标：理解推理加速、评测与量化导出的调用路径与配置项。

- 材料：examples/、docs/、swift/（推理/评测/量化相关子模块）

- 动作：

- 推理：梳理 --infer_backend (pytorch|vllm|sglang|lmdeploy) 的分发逻辑与适配层，跑通至少两种后端。

- 评测：熟悉 EvalScope 的调用方式、指标产出与数据适配。

- 量化：理解 BNB/AWQ/GPTQ/FP8 等导出路径与与推理后端的兼容性。

- 产出：一页“场景选型与配置速查表”（不同后端/量化的优缺点与开关）。

### 第7天：多模态与 RLHF（进阶）

- 目标：能跟随示例完成一次 RLHF（如 GRPO/DPO）或多模态训练。

- 材料：examples/train/rlhf/*、examples/train/multimodal/*、docs/

- 动作：

- 跑通一次小规模 RLHF（可优先 GRPO），理解奖励模型/采样/多轮设置。

- 跑通一次多模态（图像/语音）任务，掌握数据格式与处理差异。

- 产出：两段可复现实验脚本（RLHF、多模态）与记录的资源占用。

### 第8天：测试与工程化

- 目标：会读/写基础测试、按规范提交改动。

- 材料：tests/、CONTRIBUTING*.md、.github/、setup.cfg

- 动作：

- 查阅 tests/ 中与你使用的命令/模块相关的测试文件，学习最小用法。

- 补一个小测试（如：新增数据预处理单元测试）。

- 按贡献文档要求运行格式化/静态检查/单测，了解 CI 触发与检查项。

- 产出：一个最小 PR 草稿（本地分支 + 通过本地检查）。

### 第9–10天：定制与贡献（选做/整合）

- 目标：完成一次“增量型二次开发”。

- 材料：前述全部

- 动作（任选一题）：

- 新增一个数据集适配器，复用现有模型完成 SFT。

- 新增一个简单的训练技巧（如自定义回调/metric），在 examples/ 里演示。

- 将训练好的 LoRA 做量化导出并用 vLLM 或 SGLang 部署推理 API。

- 产出：可运行示例 + 文档说明（放到 examples/ 风格的结构中）。

## 学习过程中可用的“定位技巧”

- 从 setup.py 的 entry_points 精准跳到 swift/cli/ 源码。

- 以 examples/ 的命令行为“搜索关键词”，在 swift/ 里查找对应子命令的实现与参数定义。

- 优先阅读与自己刚跑过的示例同名或同路径的测试用例（tests/），理解最小可行调用栈。

## 里程碑验收

- 跑通：SFT + 推理（vLLM 或 SGLang）+ 评测（EvalScope）+ 部署（任选其一）。

- 能画出：训练主流程（CLI → 配置 → 模型/数据 → 训练/评测/保存）的模块图。

- 能提交：一个小功能或小修复的 PR（含测试与文档片段）。

- 本计划按顶层目录组织了学习顺序：先总览与环境，再跑示例，再从 setup.py 入口读 swift/ 源码，随后覆盖推理/评测/量化/多模态与工程化，最后完成一次二次开发与最小 PR。