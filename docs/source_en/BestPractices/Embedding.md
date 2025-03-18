# Embedding Training

SWIFT has already supported the training of Embedding models, including both pure text and multimodal types. The currently supported models are:

1. **ModernBERT Embedding Model**
   - [ModelScope](https://modelscope.cn/models/iic/gte-modernbert-base) | [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
2. **GTE Embedding Models**
   - **1.5B**: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-1.5B-instruct) | [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
   - **7B**: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-7B-instruct) | [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
3. **GME Embedding Models**
   - **2B**: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct) | [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)
   - **7B**: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct) | [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct)

Developers can integrate their own models independently. The `forward` output of the model needs to satisfy:

```json
{"last_hidden_state": some-embedding-tensor}
```

The return value should be a JSON with the key `last_hidden_state`, and the value should be the embedding tensor. For the input part, you can use the templates we have already supported.

**Note:** Currently, SWIFT supports embedding models that conform to pure text or multimodal LLMs. It does not support the training of CLIP-type models at this time.

Besides, All Embedding models supported by SWIFT have a normalize layer at last, consider add one when you are adding new models.

## Loss

The Embedding models supported by SWIFT currently can use the following loss functions:

- **cosine_similarity**: Cosine similarity loss, which calculates the similarity between two embeddings and fits based on the label value. It is effectively an MSE loss.
- **contrastive**: Contrastive learning loss with adjustable margin. Labels are only supported as 0 and 1.
- **online_contrastive**: Contrastive loss considering hard negatives and hard positives. Labels are only supported as 0 and 1.
- **infonce**: Computes pairwise cosine similarities between different rows within the same batch, maximizing similarity within rows and minimizing similarity between different rows. No labels are required.

The source code for the loss functions can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py).

## Dataset Format

### Format for `cosine_similarity` Loss

```json
# LLM
{"query": "sentence1", "response": "sentence2", "label": 0.8}
# MLLM
{"query": "<image>", "response": "sentence", "images": "/some/images.jpg", "label": 0.7}
{"query": "<image>sentence1", "response": "sentence2", "images": "/some/images.jpg", "label": 0.7}
```

### Format for `contrastive` / `online_contrastive` Loss

```json
# LLM
{"query": "sentence1", "response": "sentence2", "label": 1}
# MLLM
{"query": "<image>", "response": "sentence", "images": "/some/images.jpg", "label": 1}
{"query": "<image>sentence1", "response": "sentence2", "images": "/some/images.jpg", "label": 0}
```

### Format for `infonce`

```json
# LLM
{"query": "sentence1", "response": "sentence2"}
# MLLM
{"query": "<image>", "response": "sentence", "images": "/some/images.jpg"}
{"query": "<image>sentence1", "response": "sentence2", "images": "/some/images.jpg"}
```

When using a dataset in `infonce` format, please set the values of `per_device_train_batch_size` and `per_device_eval_batch_size` to greater than 1.

## Scaffolding

SWIFT provides two scaffold training scripts:

- [GTE Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gte.sh)
- [GME Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gme.sh)
