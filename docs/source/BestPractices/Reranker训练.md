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

### 1. 分类式Reranker

**适用模型：** modernbert reranker模型（如gte-reranker-modernbert-base）

**核心原理：**
- 基于序列分类架构，在预训练模型基础上添加分类头
- 输入：query-document对，输出：单个相关性分数


### 2. 生成式Reranker

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
- `LISTWISE_RERANKER_MIN_GROUP_SIZE`：最小组大小，如果组内文档数量小于该值，则不计算损失（默认：2）

**Listwise vs Pointwise：**
- **Pointwise：** 独立判断相关性，训练简单，但忽略了文档间的相对关系
- **Listwise：** 学习相对排序，性能更优，更适合排序任务的本质需求

loss的源代码可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py)找到。

## 数据集格式

```json lines
{"messages": [{"role": "user", "content": "query"}], "positive_messages": [[{"role": "assistant", "content": "relevant_doc1"}],[{"role": "assistant", "content": "relevant_doc2"}]], "negative_messages": [[{"role": "assistant", "content": "irrelevant_doc1"}],[{"role": "assistant", "content": "irrelevant_doc2"}], ...]}
```

**字段说明：**
- `messages`：查询文本
- `positive_messages`：与查询相关的正例文档列表，支持多个正例
- `negative_messages`：与查询不相关的负例文档列表，支持多个负例

**环境变量配置：**
- `MAX_POSITIVE_SAMPLES`：每个query的最大正例数量（默认：1）
- `MAX_NEGATIVE_SAMPLES`：每个query的最大负例数量（默认：7）

> 默认会从每条数据中取出`MAX_POSITIVE_SAMPLES`条正样本和`MAX_NEGATIVE_SAMPLES`条负样本，每条正样本会和`MAX_NEGATIVE_SAMPLES`条负样本组成一个group，因此每条数据会扩展成`MAX_POSITIVE_SAMPLES`x`(1 + MAX_NEGATIVE_SAMPLES)`条数据。
> 如果数据中正例/负例数量不足，会取全部正例/负例，如果数据中正例和负例数量超过`MAX_POSITIVE_SAMPLES`和`MAX_NEGATIVE_SAMPLES`，会进行随机采样。
> **IMPORTANT**：展开后的数据会放在同一个batch中，因此每个设备上的实际批处理大小（effective batch size）将是 `per_device_train_batch_size` × `MAX_POSITIVE_SAMPLES` × (1 + `MAX_NEGATIVE_SAMPLES`)。请注意调整 `per_device_train_batch_size` 以避免显存不足。

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [Pointwise分类式Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_reranker.sh)
- [Pointwise生成式Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_generative_reranker.sh)
- [Listwise分类式Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_reranker_listwise.sh)
- [Listwise生成式Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_generative_reranker_listwise.sh)

## 高级功能

- Qwen3-Reranker 自定义 Instruction：
  - 默认模板如下：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}<|im_end|>
<|im_start|>assistant
<think>

</think>


```

- 默认 Instruction：
  - `Given a web search query, retrieve relevant passages that answer the query`

- Instruction 优先级（就近覆盖）：
  - `positive_messages`/`negative_messages` 内提供的 `system` > 主 `messages` 的 `system` > 默认 Instruction。
  - 即：若某个 positive/negative 的消息序列内包含 `system`，则优先使用该条；否则若主 `messages` 含 `system` 则使用之；两者都未提供时，使用默认 Instruction。
