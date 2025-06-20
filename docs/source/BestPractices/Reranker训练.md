# Reranker训练

SWIFT已经支持Reranker模型的训练，目前已经支持的模型有：

1. modernbert reranker模型
   - [ModelScope](https://www.modelscope.cn/models/iic/gte-reranker-modernbert-base) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
2. qwen3-reranker模型
   - 0.6B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
   - 4B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
   - 8B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B) [Hugging Face](https://huggingface.co/Qwen/
   Qwen3-Reranker-8B)

## 实现方式

目前SWIFT支持两种Reranker模型的实现方式，二者在架构和损失函数计算上有显著差异：

### 1. 分类式Reranker（Classification Reranker）

**适用模型：** modernbert reranker模型（如gte-reranker-modernbert-base）

**核心原理：**
- 基于序列分类架构，在预训练模型基础上添加分类头
- 输入：query-document对，输出：单个相关性分数
- 使用BCEWithLogitsLoss进行二分类训练

### 2. 生成式Reranker（Generative Reranker）

**适用模型：** qwen3-reranker模型（0.6B/4B/8B）

**核心原理：**
- 基于生成式语言模型架构（CausalLM）
- 输入：query-document对，输出：特定token的概率（如"yes"/"no"）
- 通过对比最后位置特定token的logits进行分类

loss的源代码可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py)找到。

## 数据集格式

```json lines
{"query": "query", "positive": ["relevant_doc1", "relevant_doc2", ...], "negative": ["irrelevant_doc1", "irrelevant_doc2", ...]}
```

> 参考[MTEB/scidocs-reranking](https://www.modelscope.cn/datasets/MTEB/scidocs-reranking)

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [分类式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker.sh)
- [生成式Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker.sh)
