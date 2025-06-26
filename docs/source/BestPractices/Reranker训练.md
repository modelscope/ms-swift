# Reranker训练

SWIFT已经支持Reranker模型的训练，目前已经支持的模型有：

1. modernbert reranker模型
   - [ModelScope](https://www.modelscope.cn/models/iic/gte-reranker-modernbert-base) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
2. qwen3-reranker模型
   - 0.6B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
   - 4B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
   - 8B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

## 实现方式

目前SWIFT支持两种Reranker模型的实现方式，二者在架构和损失函数计算上有显著差异：

### 1. 分类式Reranker（Classification Reranker）

**适用模型：** modernbert reranker模型（如gte-reranker-modernbert-base）

**核心原理：**
- 基于序列分类架构，在预训练模型基础上添加分类头
- 输入：query-document对，输出：单个相关性分数


### 2. 生成式Reranker（Generative Reranker）

**适用模型：** qwen3-reranker模型（0.6B/4B/8B）

**核心原理：**
- 基于生成式语言模型架构（CausalLM）
- 输入：query-document对，输出：特定token的概率（如"yes"/"no"）
- 通过对比最后位置特定token的logits进行分类

## 损失函数类型

SWIFT支持多种损失函数来训练Reranker模型：

### Pointwise损失函数
Pointwise方法将排序问题转化为二分类问题，独立处理每个query-document对：

- **核心思想：** 对每个query-document对进行二分类，判断文档是否与查询相关
- **损失函数：** 二分类交叉熵
- **适用场景：** 简单高效，适合大规模数据训练

环境变量配置：
- `GENERATIVE_RERANKER_POSITIVE_TOKEN`：正例token（默认："yes"）
- `GENERATIVE_RERANKER_NEGATIVE_TOKEN`：负例token（默认："no"）

### Listwise损失函数
Listwise方法将排序问题转化为多分类问题，从多个候选文档中选择正例：

- **核心思想：** 对每个query的候选文档组（1个正例 + n个负例）进行多分类，识别正例文档
- **损失函数：** 多分类交叉熵
- **适用场景：** 学习文档间的相对排序关系，更符合信息检索的实际需求

环境变量配置：
- `LISTWISE_RERANKER_TEMPERATURE`：softmax温度参数（默认：1.0）
- `LISTWISE_RERANKER_MIN_GROUP_SIZE`：最小组大小（默认：2）
- `LISTWISE_GENERATIVE_RERANKER_TEMPERATURE`：listwise温度参数（默认：1.0）
- `LISTWISE_GENERATIVE_RERANKER_MIN_GROUP_SIZE`：最小组大小（默认：2）

**Listwise vs Pointwise：**
- **Pointwise：** 独立判断相关性，训练简单，但忽略了文档间的相对关系
- **Listwise：** 学习相对排序，性能更优，更适合排序任务的本质需求

## 评估指标

SWIFT为Reranker训练提供了专业的信息检索评估指标：

### MRR (Mean Reciprocal Rank)
- **定义：** 所有查询的倒数排名的平均值
- **计算方式：** MRR = (1/|Q|) × Σ(1/rank_i)，其中rank_i是第i个查询的正例文档排名
- **取值范围：** [0, 1]，越大越好
- **适用场景：** 关注正例文档在排序结果中的位置

### NDCG (Normalized Discounted Cumulative Gain)
- **定义：** 标准化折扣累积增益
- **计算方式：** NDCG = DCG / IDCG，考虑了排序位置对相关性的影响
- **取值范围：** [0, 1]，越大越好
- **适用场景：** 综合评估排序质量，对top位置的相关性更敏感

**指标计算说明：**
- 指标基于query分组计算，每个query组以正例文档开始，后跟负例文档
- 数据格式：`[1,0,0,1,0,0,0]` 表示2个query：query1=[1,0,0]，query2=[1,0,0,0]
- 自动识别query边界并分别计算每个query的指标，最后取平均值

loss的源代码可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py)找到。

## 数据集格式

```json lines
{"query": "query", "positive": ["relevant_doc1", "relevant_doc2", ...], "negative": ["irrelevant_doc1", "irrelevant_doc2", ...]}
```

> 参考[MTEB/scidocs-reranking](https://www.modelscope.cn/datasets/MTEB/scidocs-reranking)

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [Pointwise分类式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker.sh)
- [Pointwise生成式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker.sh)
- [Listwise分类式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker_listwise.sh)
- [Listwise生成式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker_listwise.sh)
