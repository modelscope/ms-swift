
# Grok 300B训练和推理实战

本文介绍了使用8卡环境对Grok-MoE 300B模型进行微调和推理的流程。


## 目录
- [环境准备](#环境准备)
- [微调](#微调)
- [推理](#推理)

## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

## 微调

### 实验环境

- GPU:8*A100 80G
- 镜像:ModelScope官方镜像1.13.1版本
- peft:0.10.0

### 数据集准备

Grok是base模型，因此我们使用了[问题生成数据集DuReader](https://www.modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)作为训练集。该数据集约15000条，max-length设置为512，训练数据约10000条（平均长度305±92 tokens）。

### 模型准备

Grok模型我们使用了[ColossalAI提供的版本](https://www.modelscope.cn/models/colossalai/grok-1-pytorch/summary)，其中我们额外准备了[符合transformers标准的tokenizer](https://www.modelscope.cn/models/AI-ModelScope/grok-1-tokenizer/summary)。

### 训练

由于Grok模型过大，device_map和deepspeed zero3非offload均无法运行训练，因此本次实验我们使用了LoRA+deepspeed zero3 offload模式运行训练。训练完整脚本如下：

```shell
# cd examples/pytorch/llm first
nproc_per_node=8

PYTHONPATH=../../.. \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_type grok-1 \
    --sft_type lora \
    --tuner_backend peft \
    --dtype bf16 \
    --output_dir output \
    --ddp_backend nccl \
    --dataset dureader-robust-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 512 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_dtype AUTO \
    --lora_target_modules DEFAULT \
    --gradient_checkpointing true \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --deepspeed zero3-offload \
```

完整的训练文件可以在[这里](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/grok-1/lora_ddp_ds)找到。

下面是训练过程的一些benchmark：

| 指标     | 值                                                           |
| -------- | ------------------------------------------------------------ |
| 显存占用 | 8*21G                                                        |
| 训练速度 | 45s/it                                                       |
| 总iter数 | 340（10000(dataset_length)/16(gradient_accumulation)/2(batch_size)） |

<img src="../../resources/image-20240329122854204.png" alt="image-20240329122854204" style="zoom: 33%;" />

由于显存占用不到24G，理论上可以在RTX3090/A10环境中运行训练。

<img src="../../resources/grok_train_loss.png" alt="train_loss (1)" style="zoom:33%;" />

<img src="../../resources/grok_train_acc.png" alt="train_acc" style="zoom:33%;" />

训练时长约4小时。

### 推理

SWIFT框架目前并不支持deepspeed推理，因此我们仍然使用transformers的device_map进行推理支持。但由于模型过大，因此部分layers会被offload到CPU上，并影响LoRA加载使推理出错，因此我们针对peft的实现进行了一定patch（原Linear在meta设备上时不迁移LoRA，并在运行时动态迁移weights）。

推理脚本如下：

```shell
# cd examples/pytorch/llm first
PYTHONPATH=../../.. \
python llm_infer.py \
    --ckpt_dir output/grok-1/vx-xxx-xxx/checkpoint-xxx \
    --dtype bf16 \
    --load_dataset_config true \
    --max_new_tokens 64 \
    --do_sample true \
    --dtype bf16 \
    --eval_human false \
    --merge_lora false \
```

推理结果：

```text
[PROMPT]Task: Question Generation
Context: 我个人感觉是吕颂贤版,剧情和原著差别不大,虽然TVB演员颜值和风光没有大陆的好。但是香港特区人口和地域的限制,只能注重在演员的演技方面发挥很出色,楼主看过大陆排《笑傲江湖》吧!在台词上表现的很生硬没有香港的注重神色配台词,比如杜燕歌把吕颂贤表情和性格几乎和原著差别不大。武打几乎沿用徐克和程小东动作的风格很注重实际技巧,没有大陆版的在武打场面依靠电脑特效表现的太夸张了。李亚鹏版的武打动作和导演还是香港的元彬,大陆毕竟还是在武侠剧起步的比较晚,主要是还是靠明星大腕压阵而香港却是恰恰相反。
Answer: 吕颂贤版
Question:[OUTPUT]笑傲江湖哪个版本好看</s>

[LABELS]笑傲江湖哪个版本好看
--------------------------------------------------
[PROMPT]Task: Question Generation
Context: 这位朋友你好,女性出现妊娠反应一般是从6-12周左右,也就是女性怀孕1个多月就会开始出现反应,第3个月的时候,妊辰反应基本结束。 而大部分女性怀孕初期都会出现恶心、呕吐的感觉,这些症状都是因人而异的,除非恶心、呕吐的非常厉害,才需要就医,否则这些都是刚怀孕的的正常症状。1-3个月的时候可以观察一下自己的皮肤,一般女性怀孕初期可能会产生皮肤色素沉淀或是腹壁产生妊娠纹,特别是在怀孕的后期更加明显。 还有很多女性怀孕初期会出现疲倦、嗜睡的情况。怀孕三个月的时候,膀胱会受到日益胀大的子宫的压迫,容量会变小,所以怀孕期间也会有尿频的现象出现。月经停止也是刚怀孕最容易出现的症状,只要是平时月经正常的女性,在性行为后超过正常经期两周,就有可能是怀孕了。 如果你想判断自己是否怀孕,可以看看自己有没有这些反应。当然这也只是多数人的怀孕表现,也有部分女性怀孕表现并不完全是这样,如果你无法确定自己是否怀孕,最好去医院检查一下。
Answer: 6-12周
Question:[OUTPUT]怀孕几个月开始反应</s>

[LABELS]怀孕多久会有反应
--------------------------------------------------
```
