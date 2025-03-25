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

> **Note:**
> 1. The `<image>` tag in the multimodal section below can appear in any position within `query`, `response`, or `rejected_response`. It is only required that the number of tags matches the number of values in `images`.
> 2. The correspondence between tags and `images` follows the order: first matching the `<image>` tags in `query`, then those in `response`, and finally parsing the `<image>` tags in `rejected_response` sequentially.
> 3. `query` represents the anchor sample, `response` represents the positive or contrastive sample, and `rejected_response` corresponds to hard negative samples.
> 4. The `<video>` and `<audio>` tags are also supported, enabling native support for video and audio embeddings.

### Format for Cosine Similarity Loss

```json lines
# LLM
{"query": "sentence1", "response": "sentence2", "label": 0.8}
# MLLM
{"query": "<image>", "response": "<image>sentence", "images": ["/some/images1.jpg", "/some/images2.jpg"], "label": 0.7}
{"query": "sentence1", "response": "<image>sentence2", "images": ["/some/images1.jpg"], "label": 0.7}
```

The eval metrics are the Pearson and Spearman's Rank Correlation Coefficient of the embeddings' euclidean distance/dot production and so on, totally 8 values.

### Format for Contrastive/Online Contrastive Loss

```json lines
# LLM
{"query": "sentence1", "response": "sentence2", "label": 1}
# MLLM
{"query": "<image>", "response": "sentence", "images": "/some/images.jpg", "label": 1}
{"query": "<image>sentence1", "response": "sentence2", "images": "/some/images.jpg", "label": 0}
```

### Format for InfoNCE

```json lines
# LLM
{"query": "sentence1", "response": "sentence2"}
# MLLM
{"query": "<image>", "response": "sentence", "images": "/some/images.jpg"}
{"query": "<image>sentence1", "response": "<image>sentence2", "rejected_response": ["<image>sentence1", "<image>sentence2"], "images": ["/some/images.jpg", "/some/images.jpg", "/some/images.jpg", "/some/images.jpg"]}
```

InfoNCE loss supports the following environment variables:
1. `INFONCE_TEMPERATURE`: The temperature parameter. If not set, the default value is 0.01.
2. `INFONCE_USE_BATCH`: Determines whether to use `rejected_response` within the sample (hard negative samples) or to use all `responses` within a batch. The default is `True`, which means using responses within the batch.
3. `INFONCE_HARD_NEGATIVES`: The number of hard negatives. If not set, all samples in `rejected_response` will be used. Since the lengths may not be consistent, a for loop will be used to compute the loss (which is slower). If set to a specific number, and there are not enough samples, the missing number will be randomly sampled. If there are excess samples, the first `INFONCE_HARD_NEGATIVES` will be selected.
4. `INFONCE_MASK_FAKE_NEGATIVE`: Masks out fake negatives. The default is set to False. When enabled, it checks if a sample's similarity is greater than the positive sample's similarity plus 0.1. If so, the sample's similarity is set to -inf to prevent the leakage of the positive sample.

> It is also possible to set the number of hard negatives to be equal in the dataset, so that even if not set, the for loop method will not be used, thereby speeding up computation.
>
> `rejected_response` can also be omitted. In this case, `INFONCE_USE_BATCH` remains `True` and will use other samples within the batch as rejected responses.

The evaluation of InfoNCE loss includes the following metrics:
- mean_neg: The average of all hard negatives
- mean_pos: The average of all positives
- margin: The average of (positive - max hard negative)

## Scaffolding

SWIFT provides two scaffold training scripts:

- [GTE Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gte.sh)
- [GME Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gme.sh)
