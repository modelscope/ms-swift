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

```json lines
{"messages": [{"role": "user", "content": "query"}], "positive_messages": [[{"role": "assistant", "content": "relevant_doc1"}],[{"role": "assistant", "content": "relevant_doc2"}]], "negative_messages": [[{"role": "assistant", "content": "irrelevant_doc1"}],[{"role": "assistant", "content": "irrelevant_doc2"}], ...]}
```

**Field Description:**
- `messages`: Query text
- `positive_messages`: List of positive documents relevant to the query, supports multiple positive examples
- `negative_messages`: List of negative documents irrelevant to the query, supports multiple negative examples

**Environment Variable Configuration:**
- `MAX_POSITIVE_SAMPLES`: Maximum number of positive examples per query (default: 1)
- `MAX_NEGATIVE_SAMPLES`: Maximum number of negative examples per query (default: 7)

> By default, `MAX_POSITIVE_SAMPLES` positive examples and `MAX_NEGATIVE_SAMPLES` negative examples will be extracted from each data item. Each positive example will be grouped with `MAX_NEGATIVE_SAMPLES` negative examples to form a group. Therefore, each data item will be expanded into `MAX_POSITIVE_SAMPLES`x`(1 + MAX_NEGATIVE_SAMPLES)` data points.
> If the number of positive/negative examples in the data is insufficient, all positive/negative examples will be used. If the number of positive and negative examples in the data exceeds `MAX_POSITIVE_SAMPLES` and `MAX_NEGATIVE_SAMPLES`, random sampling will be performed.
> **IMPORTANT**: The expanded data will be placed in the same batch. Therefore, the effective batch size on each device will be `per_device_train_batch_size` × `MAX_POSITIVE_SAMPLES` × (1 + `MAX_NEGATIVE_SAMPLES`). Please adjust your `per_device_train_batch_size` accordingly to avoid out-of-memory errors.

## Training Scripts

SWIFT provides four training script templates:

- [Pointwise Classification Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_reranker.sh)
- [Pointwise Generative Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_generative_reranker.sh)
- [Listwise Classification Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_reranker_listwise.sh)
- [Listwise Generative Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_generative_reranker_listwise.sh)

## Advanced

- Qwen3-Reranker Custom Instruction:
  - Default template:

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

- Default instruction:
  - `Given a web search query, retrieve relevant passages that answer the query`

- Instruction priority (nearest wins):
  - `system` inside `positive_messages`/`negative_messages` > `system` in main `messages` > default instruction.
  - That is, if a positive/negative message sequence contains a `system`, it takes precedence; otherwise, if main `messages` has a `system`, use it; if neither is provided, use the default instruction.
