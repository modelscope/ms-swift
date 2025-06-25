# Reranker Training

SWIFT supports Reranker model training. Currently supported models include:

1. modernbert reranker model
   - [ModelScope](https://www.modelscope.cn/models/iic/gte-reranker-modernbert-base) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
2. qwen3-reranker model
   - 0.6B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
   - 4B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
   - 8B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

## Implementation Methods

SWIFT currently supports two implementation methods for Reranker models, which have significant differences in architecture and loss function computation:

### 1. Classification Reranker

**Applicable Models:** modernbert reranker models (e.g., gte-reranker-modernbert-base)

**Core Principles:**
- Based on sequence classification architecture, adding a classification head on top of pre-trained models
- Input: query-document pairs, Output: single relevance score

### 2. Generative Reranker

**Applicable Models:** qwen3-reranker models (0.6B/4B/8B)

**Core Principles:**
- Based on generative language model architecture (CausalLM)
- Input: query-document pairs, Output: probability of specific tokens (e.g., "yes"/"no")
- Classification is performed by comparing logits of specific tokens at the final position

## Loss Function Types

SWIFT supports multiple loss functions for training Reranker models:

### Pointwise Loss Functions
Pointwise methods transform the ranking problem into a binary classification problem, processing each query-document pair independently:

- **Core Idea:** Binary classification for each query-document pair to determine document relevance to the query
- **Loss Function:** Binary cross-entropy
- **Use Cases:** Simple and efficient, suitable for large-scale data training

Environment variable configuration:
- `GENERATIVE_RERANKER_POSITIVE_TOKEN`: Positive token (default: "yes")
- `GENERATIVE_RERANKER_NEGATIVE_TOKEN`: Negative token (default: "no")

### Listwise Loss Functions
Listwise methods transform the ranking problem into a multi-classification problem, selecting positive examples from multiple candidate documents:

- **Core Idea:** Multi-classification for each query's candidate document group (1 positive + n negative examples) to identify positive documents
- **Loss Function:** Multi-class cross-entropy
- **Use Cases:** Learning relative ranking relationships between documents, better aligned with the actual needs of information retrieval

Environment variable configuration:
- `LISTWISE_RERANKER_TEMPERATURE`: Softmax temperature parameter (default: 1.0)
- `LISTWISE_RERANKER_MIN_GROUP_SIZE`: Minimum group size, if the number of documents in the group is less than this value, the loss will not be calculated (default: 2)

**Listwise vs Pointwise:**
- **Pointwise:** Independent relevance judgment, simple training, but ignores relative relationships between documents
- **Listwise:** Learning relative ranking, better performance, more suitable for the essential needs of ranking tasks

The loss function source code can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py).

## Dataset Format

### Common Original Data Format

```json lines
{"query": "query", "positive": ["relevant_doc1", "relevant_doc2", ...], "negative": ["irrelevant_doc1", "irrelevant_doc2", ...]}
```

> Reference: [MTEB/scidocs-reranking](https://www.modelscope.cn/datasets/MTEB/scidocs-reranking)

### Converted Data Format

```json lines
{"query": "query", "response": "relevant_doc1", "rejected_response": ["irrelevant_doc1", "irrelevant_doc2", ...]}
{"query": "query", "response": "relevant_doc2", "rejected_response": ["irrelevant_doc1", "irrelevant_doc2", ...]}
...
```

> The final converted data format is required, developers can build their own dataset or reuse [MTEBRerankPreprocessor](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/dataset/llm.py#L381) to convert data format.

## Training Scripts

SWIFT provides four training script templates:

- [Pointwise Classification Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker.sh)
- [Pointwise Generative Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker.sh)
- [Listwise Classification Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker_listwise.sh)
- [Listwise Generative Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker_listwise.sh)
