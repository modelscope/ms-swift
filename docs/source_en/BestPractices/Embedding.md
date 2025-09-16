# Embedding Training

SWIFT has already supported the training of embedding models, including both pure text and multimodal types. Currently supported models include:

1. modernbert embedding model
   - [ModelScope](https://modelscope.cn/models/iic/gte-modernbert-base) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
2. gte embedding models
   - 1.5B: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-1.5B-instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
   - 7B: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-7B-instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
3. gme embedding models
   - 2B: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)
   - 7B: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct)
4. qwen3-embedding models
   - 0.6B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
   - 4B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
   - 8B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

Developers can integrate their own models by ensuring the model forward output satisfies:

```text
{"last_hidden_state": some-embedding-tensor}
```

The return value should be a JSON with a `last_hidden_state` key, where the value is an embedding tensor. For the input part, you can use our already supported templates. Users can also specify the

```shell
   --task_type embedding
```
parameter to convert any other model into an embedding model for training.

It should be noted that the embedding models currently supported by SWIFT are all based on pure text or multimodal LLMs, and CLIP-type model training is not currently supported.

Additionally, all embedding models supported by SWIFT have normalization added at the end of the model forward pass. If you add new models yourself, please remember to include a normalization layer.

## Loss

The Embedding models supported by SWIFT currently can use the following loss functions:

- **cosine_similarity**: Cosine similarity loss, which calculates the similarity between two embeddings and fits based on the label value. It is effectively an MSE loss.
- **contrastive**: Contrastive learning loss with adjustable margin. Labels are only supported as 0 and 1.
- **online_contrastive**: Contrastive loss considering hard negatives and hard positives. Labels are only supported as 0 and 1.
- **infonce**: Computes pairwise cosine similarities between different rows within the same batch, maximizing similarity within rows and minimizing similarity between different rows. No labels are required.

The source code for the loss functions can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py).

## Dataset Format

> **Note:**
> 1. The `<image>` tag can appear anywhere inside `messages`/`positive_messages`/`negative_messages`. Each group has its own image fields: `images`/`positive_images`/`negative_images` to provide paths or URLs.
> 2. There is no longer any cross-field ordering requirement. Alignment rules:
>    - `images` length equals the number of `<image>` tags in `messages`.
>    - `positive_images` and `negative_images` are both list-of-list. Their outer lengths equal the lengths of `positive_messages` and `negative_messages` respectively. For each outer item, the inner list length equals the number of `<image>` tags in that message sequence.
> 3. `messages` is the anchor sample; `positive_messages` and `negative_messages` are each a list of messages (hence one more `[]`). Accordingly, `positive_images`/`negative_images` are also list-of-list and align item-by-item.
> 4. `<video>` and `<audio>` are supported as well. Follow the same rules via `videos`/`positive_videos`/`negative_videos` and `audios`/`positive_audios`/`negative_audios`.
> 5. Current constraint: the outer length of `positive_messages` must be 1 (i.e., provide exactly one positive). Accordingly, the outer length of `positive_images` must also be 1.

### Format for Cosine Similarity Loss

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 0.8}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images1.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence"}]], "positive_images": [["/some/images2.jpg"]], "label": 0.7}
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/images.jpg"]], "label": 0.7}
```

The eval metrics are the Pearson and Spearman's Rank Correlation Coefficient of the embeddings' euclidean distance/dot production and so on, totally 8 values.

### Format for Contrastive/Online Contrastive Loss

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 1}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images1.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence"}]], "positive_images": [["/some/images2.jpg"]], "label": 1}
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/images.jpg"]], "label": 0}
```

### Format for InfoNCE

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]]}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images.jpg"], "positive_messages": [[{"role": "user", "content": "sentence"}]]}
{"messages": [{"role": "user", "content": "<image>sentence1"}], "images": ["/some/images.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/positive_images.jpg"]], "negative_messages": [[{"role": "user", "content": "<image><image>sentence3"}], [{"role": "user", "content": "<image>sentence4"}]], "negative_images": [["/some/negative_images1.jpg", "/some/negative_images2.jpg"], ["/some/negative_images3.jpg"]]}
```

InfoNCE loss supports the following environment variables:
1. `INFONCE_TEMPERATURE`: The temperature parameter. If not set, the default value is 0.01.
2. `INFONCE_USE_BATCH`: Use `negative_messages` within the sample (hard negatives) or use other samples in the batch as in-batch negatives. The default is `True`, which means using in-batch negatives.
3. `INFONCE_HARD_NEGATIVES`: The number of hard negatives. If not set, all provided `negative_messages` will be used. Since the lengths may vary, a for loop will be used to compute the loss (slower). If set to a specific number, missing items will be randomly sampled, and excess items will be truncated to the first `INFONCE_HARD_NEGATIVES`.
4. `INFONCE_MASK_FAKE_NEGATIVE`: Masks out fake negatives. The default is `False`. When enabled, it checks `positive_similarity + 0.1`; any sample with similarity larger than this threshold will have its similarity set to `-inf` to prevent positive leakage.

> You can also make the number of hard negatives equal across samples in the dataset, which avoids the for-loop computation and speeds up training even if `INFONCE_HARD_NEGATIVES` is not set.
>
> `negative_messages` can be omitted. In this case, keep `INFONCE_USE_BATCH=True` to use in-batch negatives (other samples in the batch) as negatives.

The evaluation of InfoNCE loss includes the following metrics:
- mean_neg: The average of all hard negatives
- mean_pos: The average of all positives
- margin: The average of (positive - max hard negative)

## Scaffolding

SWIFT provides two scaffold training scripts:

- [GTE Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gte.sh)
- [GME Model](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gme.sh)

## Inference

SWIFT has supported the deployment of GME、GTE、Qwen3-Embedding models，please check[here](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/embedding/client.py).

You can also use the original model's code for inference:

https://www.modelscope.cn/models/iic/gte_Qwen2-7B-instruct

https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct

If you've used other models to train embedding from scratch (for example, the original `qwen2-vl` model + `--task_type embedding`), you can also use gme's inference code, but please note:

https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct/file/view/master/gme_inference.py?status=1#L111

Please modify the template here to match the model's own template to ensure the final embeddings align correctly. It's particularly important to note that the template for the gme model is different from the chatml template for the `qwen2-vl` or `qwen2.5-vl` series. In its inference code, the ending character is `<|endoftext|>` rather than `<|im_end|>`.

## Advanced

- Qwen3-Embedding Custom Instruction:
  - By default, there is no instruction; the input prompt is: `{Query}<|endoftext|>`.
  - You can add an instruction via the system message, changing the prompt to: `{Instruction} {Query}<|endoftext|>`.
  - Example:

```json lines
{"messages": [
  {"role": "system", "content": "Answer in English and list key points briefly."},
  {"role": "user", "content": "Introduce Qwen3-Embedding"}
]}
```

> Note: The Qwen3-Embedding template prepends the system content to the first user message and uses `<|endoftext|>` as the ending token.

### Before/After Examples

- Without Instruction:

  Input data (messages):

  ```json lines
  {"messages": [
    {"role": "user", "content": "What is Qwen3-Embedding?"}
  ]}
  ```

  After template conversion (actual prompt sent to the model):

  ```text
  What is Qwen3-Embedding?<|endoftext|>
  ```

- With Instruction:

  Input data (messages with system):

  ```json lines
  {"messages": [
    {"role": "system", "content": "Answer in English and list key points briefly."},
    {"role": "user", "content": "What is Qwen3-Embedding?"}
  ]}
  ```

  After template conversion (actual prompt sent to the model):

  ```text
  Answer in English and list key points briefly. What is Qwen3-Embedding?<|endoftext|>
  ```

- Positive/Negative behave the same:

  If a system message is provided within a positive/negative sequence, it is prepended to that sequence’s first user content; if no system is provided, nothing is prepended.

  Input (one positive with system, one negative without):

  ```json lines
  {
    "messages": [
      {"role": "user", "content": "Anchor"}
    ],
    "positive_messages": [[
      {"role": "system", "content": "Instruction"},
      {"role": "user", "content": "Positive"}
    ]],
    "negative_messages": [[
      {"role": "user", "content": "Negative"}
    ]]
  }
  ```

  After template conversion (actual prompts):

  ```text
  Anchor<|endoftext|>
  Instruction Positive<|endoftext|>
  Negative<|endoftext|>
  ```
