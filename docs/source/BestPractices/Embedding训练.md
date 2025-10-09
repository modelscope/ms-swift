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
4. qwen3-embedding模型
   - 0.6B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
   - 4B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
   - 8B: [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B) [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

开发者可以自行集成自己的模型，模型forward输出值需要满足：

```text
{"last_hidden_state": some-embedding-tensor}
```

返回值是一个json，具有`last_hidden_state` key，value是embedding tensor即可，输入部分可以使用我们已经支持的template。用户也可以通过指定

```shell
   --task_type embedding
```
参数来将任意一个其他模型转换为embedding模型进行训练。

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
> 1. `<image>`标签可以出现在`messages`/`positive_messages`/`negative_messages`的任意位置；它们各自拥有独立的`images`/`positive_images`/`negative_images`字段用于提供图片路径或URL。
> 2. 不再需要跨字段的“对应顺序”。对齐规则为：`images`的长度等于`messages`中`<image>`标签的数量；`positive_images`与`negative_images`均为“list of list”，其外层长度分别等于`positive_messages`与`negative_messages`的长度；并且外层每一项的内层列表长度等于该条消息序列中`<image>`标签的数量。
> 3. `messages`代表anchor样本（anchor sample）；`positive_messages`/`negative_messages`为“list of messages”（因此多一层`[]`）；相应地，`positive_images`/`negative_images`也多一层`[]`并与之逐项对齐。
> 4. 也支持`<video>`, `<audio>`标签；可按相同规则分别通过`videos`/`positive_videos`/`negative_videos`与`audios`/`positive_audios`/`negative_audios`提供对应模态数据。
> 5. 当前约束：`positive_messages`的外层长度必须为1（即仅提供一个positive样本）；对应地，`positive_images`的外层长度也必须为1。

### cosine_similarity loss对应的格式

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 0.8}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images1.jpg"],"positive_messages": [[{"role": "user", "content": "<image>sentence"}]], "positive_images": [["/some/images2.jpg"]], "label": 0.7}
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/images.jpg"]], "label": 0.7}
```


### contrastive/online_contrastive loss对应的格式

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 1}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images1.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence"}]], "positive_images": [["/some/images2.jpg"]], "label": 1}
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/images.jpg"]], "label": 0}
```

评测的指标分别是两个embedding的欧式距离、点积等的pearson系数以及spearman系数，共八个指标。

### infonce 格式

```json lines
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]]}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images.jpg"], "positive_messages": [[{"role": "user", "content": "sentence"}]]}
{"messages": [{"role": "user", "content": "<image>sentence1"}], "images": ["/some/images.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence2"}]], "positive_images": [["/some/positive_images.jpg"]], "negative_messages": [[{"role": "user", "content": "<image><image>sentence3"}], [{"role": "user", "content": "<image>sentence4"}]], "negative_images": [["/some/negative_images1.jpg", "/some/negative_images2.jpg"], ["/some/negative_images3.jpg"]]}
```

infonce loss支持几个环境变量：
1. INFONCE_TEMPERATURE temperature参数，不设置的话默认值是0.01
2. INFONCE_USE_BATCH 使用sample内部的`negative_messages`（hard negative样例）还是使用一个batch内其他样本作为in-batch negatives；默认为True，表示使用batch内部的样本作为负例
3. INFONCE_HARD_NEGATIVES hard negatives的数量；如果不设置会使用数据中提供的所有`negative_messages`。由于长度未必一致，因此会采用for循环计算loss（计算会慢）。若设置为某个数值，则不足会随机采样补齐，超长会选用前`INFONCE_HARD_NEGATIVES`个
4. INFONCE_MASK_FAKE_NEGATIVE mask掉假negative。默认为False，开启时会判断positive sample的similarity+0.1，比该值大的sample的similarity会被设置为-inf，防止positive sample泄露问题

> 也可以在数据集中将hard negatives数量设置为数量相等，这样即使不设置也不会使用for循环方式，加快计算速度
> `negative_messages`也可以不提供。在这种情况下，保持`INFONCE_USE_BATCH=True`，会使用一个batch内部的其他样本作为负例

infonce loss的评测会有下面几个指标：
- mean_neg 所有hard_negative的平均值
- mean_pos 所有positive的平均值
- margin positive-max_hard_negative的平均值

## 脚手架

SWIFT提供了两个脚手架训练脚本：

- [gte模型](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gte.sh)
- [gme模型](https://github.com/tastelikefeet/swift/blob/main/examples/train/embedding/train_gme.sh)

## 推理

SWIFT已经支持GME、GTE、Qwen3-Embedding模型的部署，请查看[这里](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/embedding/client.py).

也可以使用原模型的代码进行推理：

https://www.modelscope.cn/models/iic/gte_Qwen2-7B-instruct

https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct

如果使用了其他模型从0训练embedding（例如，原版`qwen2-vl`模型+`--task_type embedding`），也可以使用gme的推理代码，但请注意：

https://www.modelscope.cn/models/iic/gme-Qwen2-VL-7B-Instruct/file/view/master/gme_inference.py?status=1#L111

这里的模板请修改为模型自身的template，以免最后的embedding对不上。需要额外注意的是，gme模型的template和`qwen2-vl`或`qwen2.5-vl`系列的chatml template并不相同，其推理代码最后的结束字符是`<|endoftext|>`而非`<|im_end|>`.

## 高级功能

- Qwen3-Embedding 自定义 Instruction：
  - 默认无 Instruction，输入模板为：`{Query}<|endoftext|>`。
  - 通过在 system message 中添加 Instruction，可将输入改为：`{Instruction} {Query}<|endoftext|>`。
  - 示例：

```json lines
{"messages": [
  {"role": "system", "content": "请用中文回答，并输出简洁要点"},
  {"role": "user", "content": "介绍一下Qwen3-Embedding"}
]}
```

> 说明：Qwen3-Embedding 模板会将 system 内容前置拼接到首条 user 消息中，并使用 `<|endoftext|>` 作为结束标记。

### 转换前后示例

- 不加 Instruction：

  输入数据（messages）：

  ```json lines
  {"messages": [
    {"role": "user", "content": "北京明天天气如何？"}
  ]}
  ```

  模板转换后（送入模型的实际文本）：

  ```text
  北京明天天气如何？<|endoftext|>
  ```

- 加 Instruction：

  输入数据（messages，包含system）：

  ```json lines
  {"messages": [
    {"role": "system", "content": "请使用中文、精炼输出要点"},
    {"role": "user", "content": "北京明天天气如何？"}
  ]}
  ```

  模板转换后（送入模型的实际文本）：

  ```text
  请使用中文、精炼输出要点 北京明天天气如何？<|endoftext|>
  ```

- positive/negative 同理：

  若在某个 positive/negative 的消息序列中提供 system，则会将该 system 内容前置到该序列首条 user 内容之前；未提供 system 则不前置。

  输入数据（包含一个 positive 带 system，和一个 negative 无 system）：

  ```json lines
  {
    "messages": [
      {"role": "user", "content": "Anchor"}
    ],
    "positive_messages": [[
      {"role": "system", "content": "指令"},
      {"role": "user", "content": "Positive"}
    ]],
    "negative_messages": [[
      {"role": "user", "content": "Negative"}
    ]]
  }
  ```

  模板转换后（送入模型的实际文本）：

  ```text
  Anchor<|endoftext|>
  指令 Positive<|endoftext|>
  Negative<|endoftext|>
  ```
