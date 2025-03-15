# Embedding训练

SWIFT已经支持Embedding模型的训练，包括纯文本和多模态两个类型。目前已经支持的模型有：

1. modernbert embedding模型
   - [ModelScope](https://modelscope.cn/models/iic/gte-modernbert-base) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
2. gte embedding模型
   - 1.5B: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-1.5B-instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)
   - 7B: [ModelScope](https://www.modelscope.cn/models/iic/gte_Qwen2-7B-instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)
3. gme embedding模型
   - 2B: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-2B-Instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)
   - 7B: [ModelScope](https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct) [Hugging Face](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-7B-Instruct)

开发者可以自行集成自己的模型，模型forward输出值需要满足：

```json
{"last_hidden_state": some-embedding-tensor}
```

返回值是一个json，具有`last_hidden_state` key，value是embedding tensor即可，输入部分可以使用我们已经支持的template。

需要注意的是，SWIFT目前支持的embedding模型均为符合纯文本或多模态LLM，目前并不支持CLIP类型的模型训练。

## loss

目前SWIFT支持的Embedding模型可以使用的loss有：

- cosine_similarity: cosine相似度loss，计算两个embedding的相似度，并根据label的值拟合，实际为MSE loss
- contrastive: 可调margin的对比学习loss，label仅支持0和1两个值
- online_contrastive: 考虑hard negative和hard positive部分的contrastive loss，label仅支持0和1两个值
- infonce: 在同一个batch中不同row两两计算cosine相似度，并使row内部相似度最大，不同row相似度最小，不需要label

loss的源代码可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py)找到。

## 数据集格式

### cosine_similarity loss对应的格式

```json lines
# LLM
{"query": "sentence1", "response":  "sentence2", "label": 0.8}
# MLLM
{"query": "<image>", "response":  "sentence", "images": "/some/images.jpg", "label": 0.7}
{"query": "<image>sentence1", "response":  "sentence2", "images": "/some/images.jpg", "label": 0.7}
```


### contrastive/online_contrastive loss对应的格式

```json lines
# LLM
{"query": "sentence1", "response":  "sentence2", "label": 1}
# MLLM
{"query": "<image>", "response":  "sentence", "images": "/some/images.jpg", "label": 1}
{"query": "<image>sentence1", "response":  "sentence2", "images": "/some/images.jpg", "label": 0}
```

### infonce 格式

```json lines
# LLM
{"query": "sentence1", "response":  "sentence2"}
# MLLM
{"query": "<image>", "response":  "sentence", "images": "/some/images.jpg"}
{"query": "<image>sentence1", "response":  "sentence2", "images": "/some/images.jpg"}
```

使用infonce格式的数据集请将per_device_train_batch_size和per_device_eval_batch_size的值设置为>1

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [gte模型](../../../examples/train/embedding/train_gte.sh)
- [gme模型](../../../examples/train/embedding/train_gme.sh)

