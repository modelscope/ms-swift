# LLM量化文档
swift支持使用awq, gptq, bnb, hqq, eetq技术对模型进行量化. 其中awq, gptq量化技术支持vllm进行推理加速, 且量化后的模型支持qlora微调.

**注意** 量化在不同指令下的作用不同
- sft lora训练中指定量化用于`qlora`，用于降低训练所需显存
- export中指定量化用于量化模型并保存。
- infer中指定量化用于量化模型并推理。

其中bnb,hqq,eetq无需校准数据，量化速度较快，在 sft lora 训练 和 infer 中使用，指定`--quant_method bnb/hqq/eetq`

awq,gptq需要校准数据，在 export 中使用，`--quant_method awq/gptq`

## 目录
- [环境准备](#环境准备)
- [量化微调(qlora)](#量化微调(qlora))
- [原始模型](#原始模型)
- [微调后模型](#微调后模型)
- [推送模型](#推送模型)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 使用awq量化:
# autoawq和cuda版本有对应关系，请按照`https://github.com/casper-hansen/AutoAWQ`选择版本
pip install autoawq -U

# 使用gptq量化:
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 使用hqq量化：
# 需要transformers版本>4.40，从源码安装
pip install git+https://github.com/huggingface/transformers
pip install hqq
# 如果要兼容训练，需要从源码安装peft
pip install git+https://github.com/huggingface/peft.git

# 使用eetq量化：
# 需要transformers版本>4.40，从源码安装
pip install git+https://github.com/huggingface/transformers
# 参考https://github.com/NetEase-FuXi/EETQ
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ/
git submodule update --init --recursive
pip install .
# 如果要兼容训练，需要从源码安装peft
pip install git+https://github.com/huggingface/peft.git

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 量化微调(qlora)
在sft lora训练中指定`--quant_method`和`--quantization_bit`来执行qlora，显著减少训练所需显存
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method hqq \
    --quantization_bit 4 \

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method eetq \
    --dtype fp16 \

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method bnb \
    --quantization_bit 4 \
    --dtype fp16 \
```
**注意**
- hqq支持更多自定义参数，比如为不同网络层指定不同量化配置，具体请见[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md)
- eetq量化为8bit量化，无需指定quantization_bit。目前不支持bf16，需要指定dtype为fp16
- eetq目前qlora速度比较慢，推荐使用hqq。参考[issue](https://github.com/NetEase-FuXi/EETQ/issues/17)

## 原始模型
使用bnb,hqq,eetq量化模型并推理
```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method bnb \
    --quantization_bit 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method hqq \
    --quantization_bit 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method eetq \
    --dtype fp16
```

这里展示对qwen1half-7b-chat进行awq, gptq量化.
```bash
# awq-int4量化 (使用A100大约需要18分钟, 显存占用: 13GB)
# 如果出现量化的时候OOM, 可以适度降低`--quant_n_samples`(默认256)和`--quant_seqlen`(默认2048).
# gptq-int4量化 (使用A100大约需要20分钟, 显存占用: 7GB)

# awq: 使用`alpaca-zh alpaca-en sharegpt-gpt4-mini`作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method awq

# gptq: 使用`alpaca-zh alpaca-en sharegpt-gpt4-mini`作为量化数据集
# gptq量化请先查看此issue: https://github.com/AutoGPTQ/AutoGPTQ/issues/439
OMP_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method gptq

# awq: 使用自定义量化数据集
# gptq同理
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset xxx.jsonl \
    --quant_method awq

# 推理 swift量化产生的模型
# awq
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-awq-int4
# gptq
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-gptq-int4

# 推理 原始模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

**量化效果对比**:

```python
# swift量化产生的awq-int4模型
"""
<<< 你好
你好！有什么问题我可以帮助你吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。闰年规则是：普通年份能被4整除但不能被100整除，或者能被400整除的年份就是闰年。2000年满足后者，所以按照公历是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江美食非常丰富，以下列举一些具有代表性的：

1. **杭州菜**：如西湖醋鱼、东坡肉、龙井虾仁、叫化童鸡等，清淡而精致。
2. **宁波菜**：如红烧肉、汤圆、水磨年糕、宁波汤包等，口味鲜美。
3. **温州菜**：温州海鲜如蝤蛑、蝤蛑炒年糕、瓯菜，口味偏咸鲜。
4. **绍兴菜**：如东坡肉、霉干菜扣肉、绍兴黄酒等，酒香浓郁。
5. **嘉兴粽子**：如嘉兴肉粽，特别是五芳斋的粽子，口感软糯。
6. **金华火腿**：浙江特产，以金华地区最为著名，口感醇厚。
7. **浙东土菜**：如农家小炒、野菜、腌菜等，原汁原味。

各地还有许多特色小吃，如杭州的知味观、宋城的宋嫂鱼羹，以及各种小吃街如河坊街、西湖醋鱼一条街等，都值得一尝。如果你有具体地方或口味偏好，可以告诉我，我可以给出更具体的建议。
"""

# swift量化产生的gptq-int4模型
"""
<<< 你好
你好！很高兴为你提供帮助。有什么问题或需要咨询的吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。根据格里高利历（公历），闰年的规则是：普通年份能被4整除但不能被100整除，或者能被400整除的年份就是闰年。2000年满足后两个条件，所以是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江美食丰富多样，以下是一些具有代表性的：

1. **杭州菜**：如东坡肉、西湖醋鱼、龙井虾仁、叫化鸡等，注重原汁原味，讲究刀工和火候。
2. **宁波的海鲜**：如宁波汤圆、海鲜大餐，宁波炒肉丝也很有名。
3. **嘉兴粽子**：以嘉兴五芳斋的粽子最为人知，有各种口味如肉粽、豆沙粽等。
4. **绍兴黄酒**：搭配当地特色菜肴，如醉鸡、糟溜鱼片等。
5. **金华火腿**：浙江特产，口感醇厚，常作为节日礼品。
6. **温州瓯菜**：口味偏酸辣，如温州鱼生、白斩鸡等。
7. **浙东佛跳墙**：类似福建的佛跳墙，用多种海鲜和山珍炖制。

如果你对某种具体的食物感兴趣，可以告诉我，我可以提供更详细的推荐。
"""

# 原始模型
"""
<<< 你好
你好！有什么问题我可以帮助你吗？
--------------------------------------------------
<<< 2000年是闰年嘛？
是的，2000年是闰年。根据格里高利历（公历），闰年的规则是：普通年份能被4整除但不能被100整除，或者能被400整除的年份都是闰年。2000年满足后者，所以是闰年。
--------------------------------------------------
<<< 15869+587=?
15869 + 587 = 16456
--------------------------------------------------
<<< 浙江的省会在哪
浙江省的省会是杭州市。
--------------------------------------------------
<<< 这有什么好吃的
浙江的美食非常丰富，以下列举一些具有代表性的：

1. **杭州菜**：如西湖醋鱼、东坡肉、龙井虾仁、叫化童鸡等，清淡鲜美，注重原汁原味。
2. **宁波菜**：如宁波汤圆、红烧肉、海鲜类，如宁波海鲜面、清蒸河鳗等。
3. **绍兴菜**：如霉干菜扣肉、茴香豆、醉排骨，特色是酱香浓郁。
4. **温州菜**：如温州鱼丸、白斩鸡、楠溪江三鲜，口味偏咸鲜。
5. **嘉兴粽子**：特别是嘉兴五芳斋的粽子，闻名全国，甜咸皆有。
6. **金华火腿**：浙江名特产，口感鲜美，营养丰富。
7. **浙东土菜**：如东阳火腿、嵊州菜、台州海鲜等，地方特色鲜明。

当然，浙江各地还有许多特色小吃，如衢州的鸭头、湖州的粽子、舟山的海鲜等，你可以根据自己的口味选择。如果你需要更具体的推荐，可以告诉我你对哪种类型或者哪个地方的美食感兴趣。
"""
```


## 微调后模型

假设你使用lora微调了qwen1half-4b-chat, 模型权重目录为: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

这里只介绍使用awq技术对微调后模型进行量化, 如果要使用gptq技术量化, 同理.

**Merge-LoRA & 量化**
```shell
# 使用`alpaca-zh alpaca-en sharegpt-gpt4-mini`作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method awq

# 使用微调时使用的数据集作为量化数据集
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --load_dataset_config true --quant_method awq
```

**推理量化后模型**
```shell
# awq/gptq量化模型支持vllm推理加速. 也支持模型部署.
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-awq-int4'
```

**部署量化后模型**

服务端:

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-awq-int4'
```

测试:
```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen1half-4b-chat",
"messages": [{"role": "user", "content": "晚上睡不着觉怎么办？"}],
"max_tokens": 256,
"temperature": 0
}'
```


## 推送模型
假设你使用lora微调了qwen1half-4b-chat, 模型权重目录为: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

```shell
# 推送原始量化模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-gptq-int4 \
    --push_to_hub true \
    --hub_model_id qwen1half-7b-chat-gptq-int4 \
    --hub_token '<your-sdk-token>'

# 推送lora增量模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>'

# 推送merged模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true

# 推送量化后模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true \
    --quant_bits 4
```
