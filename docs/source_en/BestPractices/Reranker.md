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
- Uses BCEWithLogitsLoss for binary classification training

### 2. Generative Reranker

**Applicable Models:** qwen3-reranker models (0.6B/4B/8B)

**Core Principles:**
- Based on generative language model architecture (CausalLM)
- Input: query-document pairs, Output: probability of specific tokens (e.g., "yes"/"no")
- Classification is performed by comparing logits of specific tokens at the final position

The loss function source code can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py).

## Dataset Format

```json lines
{"query": "query", "positive": ["relevant_doc1", "relevant_doc2", ...], "negative": ["irrelevant_doc1", "irrelevant_doc2", ...]}
```

> Reference: [MTEB/scidocs-reranking](https://www.modelscope.cn/datasets/MTEB/scidocs-reranking)

## Training Scripts

SWIFT provides two training script templates:

- [Classification Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_reranker.sh)
- [Generative Reranker](https://github.com/tastelikefeet/swift/blob/main/examples/train/reranker/train_generative_reranker.sh) 