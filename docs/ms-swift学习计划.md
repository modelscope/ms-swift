# 目标

三阶段：

- 第一阶段：能跑通示例、写基础测试并符合项目贡献规范。

- 第二阶段：熟练使用 swift 进行训练、推理、评测与部署。

- 第三阶段：理解 swift/ 核心代码结构与数据/模型/训练管道，能做二次开发（加模型/数据集/算法）。
- 第四阶段：如臂使指，熟悉swift的每一个细节，能够根据实际需求进行扩展。



# 顺序

对应顶层目录/文件的学习顺序：

1) README_CN.md/README.md（总览与能力边界，使用中文版）
2) docs/（阅读相关文档：微调、RLHF、量化、vLLM/SGLang/LmDeploy、评测 EvalScope、UI）
3) requirements/ + requirements.txt（依赖与可选组件）
4) Makefile、scripts/、.dev_scripts/、.pre-commit-config*.yaml（本地常用命令与工程规范）
5) examples/（先跑通：训练/推理/评测/部署）
6) setup.py → 定位入口：swift/cli/main.py、swift/cli/_megatron/main.py
7) swift/（源码主线：CLI → 配置解析 → 模型/数据加载 → 训练循环 → 加速/量化/评测/部署）
8) docs/（深挖：微调、RLHF、量化、vLLM/SGLang/LmDeploy、评测 EvalScope、UI）
9) tests/（验证思路与用法，学习最小可复现实例）
10) CONTRIBUTING*.md、CODE_OF_CONDUCT.md、.github/（提交流程与质量保障）
11) setup.cfg、MANIFEST.in、LICENSE、asset/（工程化与发布收尾）



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