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

此外，SWIFT支持的所有embedding模型在模型forward最后都增加了normalize，如自行增加新模型请注意增加normalize层。

## loss

目前SWIFT支持的Embedding模型可以使用的loss有：

- cosine_similarity: cosine相似度loss，计算两个embedding的相似度，并根据label的值拟合，实际为MSE loss
- contrastive: 可调margin的对比学习loss，label仅支持0和1两个值
- online_contrastive: 考虑hard negative和hard positive部分的contrastive loss，label仅支持0和1两个值
- infonce: 在同一个batch中不同row两两计算cosine相似度，并使row内部相似度最大，不同row相似度最小，不需要label

loss的源代码可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py)找到。

## 数据集格式

> 注：
> 1. 下面的多模态部分<image>标签可以出现在query/response/rejected_response的任意位置，只需要标签数量和images的值数量相等即可
> 2. 标签和images的对应顺序为先对应query中的<image>标签，然后是response中的，之后按顺序解析rejected_response中的
> 3. query代表anchor sample，response代表positive sample或对比sample，rejected_response是hard negative samples
> 4. 也支持<video>, <audio>标签，即天然支持video和audio的embedding

### cosine_similarity loss对应的格式

```json lines
# LLM
{"query": "sentence1", "response":  "sentence2", "label": 0.8}
# MLLM
{"query": "<image>", "response":  "<image>sentence", "images": ["/some/images1.jpg", "/some/images2.jpg"], "label": 0.7}
{"query": "sentence1", "response":  "<image>sentence2", "images": ["/some/images1.jpg"], "label": 0.7}
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
{"query": "<image>sentence1", "response":  "<image>sentence2", "rejected_response": ["<image>sentence1", "<image>sentence2"], "images": ["/some/images.jpg", "/some/images.jpg", "/some/images.jpg", "/some/images.jpg"]}
```

infonce loss支持几个环境变量：
1. INFONCE_TEMPERATURE temperature参数，不设置的话默认值是0.01
2. INFONCE_USE_BATCH 使用sample内部的rejected_response（hard negative样例）还是使用一个batch的所有responses，默认为True代表使用batch内部的responses
3. INFONCE_HARD_NEGATIVES hard negatives的数量，如果不设置会使用rejected_response的所有samples，由于长度未必一致，因此会采用for循环计算loss（计算会慢），如果设置为某个数值，则如果不够会对缺失数量进行随机采样，超长会选用前`INFONCE_HARD_NEGATIVES`个
4. INFONCE_MASK_FAKE_NEGATIVE mask掉假negative。默认为False，开启时会判断positive sample的similarity+0.1，比该值大的sample的similarity会被设置为-inf，防止positive sample泄露问题

> 也可以在数据集中将hard negatives数量设置为数量相等，这样即使不设置也不会使用for循环方式，加快计算速度
> rejected_response也可以没有，这种情况下INFONCE_USE_BATCH保持为True，会使用一个batch内部的其他samples作为rejected responses

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [gte模型](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gte.sh)
- [gme模型](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gme.sh)
