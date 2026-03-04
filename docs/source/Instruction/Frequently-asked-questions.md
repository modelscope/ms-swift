# 常见问题整理

下面是SWIFT使用过程中遇到的一些常见问题。

## 训练

SWIFT支持的训练方法包括预训练、指令监督微调、偏好学习、GRPO、Embedding、Reranker、序列分类任务等，详见[主页](https://github.com/modelscope/ms-swift/blob/main/README_CN.md)。

### Q1: SWIFT支持的模型有哪些？如何设置本地模型路径？
支持的模型详见文档[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)。如果模型已经下载到了本地，设置`--model <path_to_model>`即可。对于离线环境训练，同时设置`--model 本地路径`，`--check_model false`，如果提示git clone相关报错，需要clone repo，然后通过`local_repo_path`指定，详见[命令行参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。从ModelScope下载的模型，可以配置环境变量`MODELSCOPE_CACHE=your_path`将原始的模型存到指定路径；如果用ModelScope SDK下载，通过`cache_dir="本地地址"`；也可以使用`modelscope download`命令行工具或`git`下载，详见modelscope文档[模型下载](https://modelscope.cn/docs/models/download)。如果需要从Hugging Face下载模型，设置环境变量`USE_HF=1`。
SWIFT会自动匹配model_type，也可以查看文档[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)，手动指定。

### Q2: SWIFT支持的数据集有哪些？如何使用自定义数据集？
支持的数据集详见文档[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)。自定义数据集格式及使用方法见文档[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)，符合这些格式的数据集会自动使用swift内置的数据预处理器。如果与文档中的格式不一致，请自行转换格式，或者参考已支持的数据集接入。若自定义数据集中有额外的字段，这些字段默认不会被使用，可以通过[命令行参数remove_unused_columns](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id4)进行设置。
需要将数据集下载到本地，然后通过路径指定，请查看[自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dataset-info-json)。`git clone`下载到本地，然后通过dataset_info.json文件中的`dataset_path`字段指定就行。
数据随机详见[命令行参数dataset_shuffle](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。
强制重新下载数据集，设置命令行参数`--download_mode`。对数据集进行错误检查，请设置命令行参数`strict`。需要数据集质检工具时，可以查看另一个库[data-juicer](https://github.com/modelscope/data-juicer)。
由于datasets的底层pyarrow对于类型管控比较严格，图像grounding数据集的objects部分、agent数据集的tools部分等，因为这个原因要用str，要不pyarrow就会报错：你每行的类型不一致。
训练中遇到报错`AttributeError:’TrainerState’ object has no attribute ’last_model_checkpoint’`，数据集太少了，数据数量不足一个step导致的报错，增加一些数据。另外，切分的验证集数据很少时也会有类似报错。
下面是一个assistant字段为空导致的报错：
```text
File "/mnt/workspace/swift/swift/1lm/dataset/preprocessor/core. py", line 69, in _check_messages raise
ValueError(f'assistant_message; {assistant_message}')
ValueError: assistant_message: {'role' :'assistant', 'content': ''}
```
```shell
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MAX_PIXELS=1003520 swift sft --model Qwen/Qwen2.5-VL-7B-Instruct --tuner_type lora --dataset /mnt/workspace/data.json --deepspeed zero2 --max_length 16384
```
数据集assistant字段为空，如果是推理，把这个空字符串删掉，因为这个会导致训练时NaN，会做检查。

### Q3: 从缓存加载数据集相关问题
设置命令行参数`--load_from_cache_file true`，可以加快数据集加载速度，尤其是在多模态数据集、数据量较大等场景。在debug或修改preprocessor时，设置为false，更多说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索该参数。

### Q4: 如何搭建SWIFT环境？有镜像可以使用吗？
环境搭建详见[SWIFT安装文档](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)，一些常见依赖的推荐版本可以在[主页](https://github.com/modelscope/ms-swift/blob/main/README_CN.md)上找到。文档中提供了镜像，用`docker run`命令启动容器即可，如：`docker run --gpus all -p 8000:8000 -it -d --name ms modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5 /bin/bash`，启动容器后拉最新代码安装swift。

### Q5: 多模态模型训练数据格式、参数冻结、优化器设置相关问题
多模态模型训练的[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal)。支持纯文本、图文数据训练，也可以两种数据混合训练。图像、视频、音频相关的参数，如，最大像素、fps等请查看[特定模型参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id19)。
Grounding任务中通用数据格式支持了一个物体对应多个bbox，参考文档[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#grounding)。videos可以是图片列表，使用文件目录的方式。
SWIFT按max_pixels对图像进行调整，会保存预处理前和后的图像，然后对bbox进行调整，不过推理没有这样的调整，需要提前手动处理图像。
VLM模型训练减少显存使用，请配置`--freeze_vit true`，以及限制最大像素的参数`--max_pixels`。`--freeze_vit`，`--freeze_aligner`，`--freeze_llm`这几个参数详见[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#tuner)。如果ViT没有训练，那有会有warning: none of the inputs have requires_grad=True是正常的，如果训练了，则不应该抛出。
使用全参数微调visual encoder同时使用LoRA微调LLM，参考这里[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit)。

### Q6: template相关问题
由于jinja chat template没有labels，所以不支持训练。
多模态数据集如果需要在加载数据之后做动态数据增强，例如，给输入数据随机添加噪声等，请在template中修改encode方法。

### Q7: SWIFT训练如何debug？
详见[预训练与微调文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Pre-training-and-Fine-tuning.html)。

### Q8: SWIFT如何使用python脚本训练？
参考[notebook例子](https://github.com/modelscope/ms-swift/tree/main/examples/notebook)。

### Q9: SWIFT如何使用UI界面训练？
使用`swift web-ui`命令，界面训练与命令行一致，界面上的参数请查看命令行参数文档。自定义数据集的使用与上面Q2一致。Megatron-SWIFT不支持UI界面训练。

### Q10: 单机多卡训练相关问题
SWIFT多卡训练底层依赖torchrun。`deepspeed` 和 `device_map`不兼容，两个只能选1个。更多细节请查看代码库中的[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu)。

### Q11: 多机多卡训练相关问题
请查看[多机多卡例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)。多机多卡训练时，只有主节点有日志。
多机训练速度缓慢，如，使用DeepSpeed ZeRO3训练会出现严重的速度下降，请查看[issue](https://github.com/modelscope/ms-swift/issues/1825)。

### Q12: 大规模数据集相关问题
数据集太大了，然后每次tokenize都需要很久，请使用`lazy_tokenize`或流式读取`streaming`，详见[命令行参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。

### Q13: 断点续训相关问题
先前训练脚本中的参数不变，加上`--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`，详见[命令行参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。如果数据集发生了改动，仅加载模型，请同时设置`--resume_only_model`。更复杂的场景，请在命令行参数文档中搜索resume。

### Q14: 数据集流式加载相关问题
流式加载`--streaming true`，一边训练一边加载，需要设置max_steps，详见`streaming`参数说明，[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id4)。
注意：streaming是不随机的，也不划分验证集，验证集通过命令行参数`val_dataset`指定。
断点续训时，流式只能往前索引，不能随机索引，跳过已经训练的数据耗时特别长，不建议用流式。

### Q15: packing相关问题
packing要和flash_attn一起使用，不然是有误差，attention_mask会出问题。packing_cache这个参数，在多机训练时，需要设置为共享的磁盘路径。
Qwen3.5模型中的linear-attention不支持var_len，不建议开启packing。
开启packing，多模态数据会有两次map，map完一次后还会进行第二次mapping，一次是数据集的，一次是template的。如果速度非常慢，可以设置`OMP_NUM_THREADS=14`加速，或者可以把packing去掉，就不会有第二次了。

### Q16: 数据集多进程处理
数据集map过程比较慢时，设置参数`--dataset_num_proc`可以开多进程。多模态数据集map比较慢是正常的。

### Q17: 当前训练完默认保存多少个checkpoint？
默认保存所有的checkpoint，详见[命令行参数 save_total_limit](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。

### Q18: 训练过程的loss和acc
自定义的损失函数在plugin中加就可以。如果需要不同数据集的loss曲线，请设置`--enable_channel_loss`。
如果eval得到的acc和对应保存的ckpt去重新推理一遍计算得到的acc不是一致的，是因为训练时候的eval_acc和推理时候的acc计算方式不一样导致的。`acc_strategy`: 默认为`'token'`, 可选择的值包括: `'token'`，`'seq'`。
训练过程中没有token_acc是因为有些模型`logits`和`labels`数量对不上，就不算的。
可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py)查看当前支持的loss或添加新的loss，
检查`<image>`等特殊token是否参与损失计算，可以在命令行日志中找一下打印的labels。
训练agent时，tool_call就是应该算loss，tool_response不算loss。

### Q19: 模型参数freeze相关问题
训练的过程中，冻结某些层时导致某些参数未参与梯度回传，请配置参数`--ddp_find_unused_parameters true`。
freeze_parameters和freeze_vit/freeze_aligner/freeze_llm：先freeze parameters再active parameters。`freeze vit/freeze aligner/freeze llm`这三个参数会对freeze parameters 和trainable parameters进行调整.因为有些模型的ViT中包含`aligner`，所以会将`aligner`单独加入trainable_parameters。
freeze_parameters_ratio这个参数的机制是从embedding开始从下往上freeze。

### Q20: 序列并行相关问题
序列并行支持pt, sft, dpo and grpo。参考这个例子[sequence_parallel](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel)。
VLM模型的目前仅支持flash-attn，纯文本支持flash-attn和sdpa。
sequence parallel可以和Liger kernel同时使用。
sequence parallel和自定义loss冲突时，由于sequence parallel在自己的代码中定制了loss，可以自己改下[这里](https://github.com/modelscope/ms-swift/blob/main/swift/trainers/sequence_parallel/ulysses.py)。

### Q21: 扩充词表
用SWIFT框架扩充词表需要设置命令行参数`new_special_tokens`，`--modules_to_save embed_tokens lm_head`，详见[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens)。

### Q22: tuners相关问题
SWIFT中的LlamaPro对多模态做了适配。
LongLoRA只有LLaMA系列模型能用。
LoRA训练和`--trainable_parameters`参数不兼容，LoRA模块之外其他的可训练参数用modules_to_save。

### Q23: embedding/reranker训练
[embedding训练例子](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding)。
[reranker训练例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/reranker)。
数据格式见[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)。

### Q24: 分类任务训练
SWIFT支持多标签分类，自定义数据集文档有数据格式，在命令行参数文档中搜索`problem_type`，其他和回归是一样的。
注意：label字段和message字段同级。

### Q25: thinking模型训练
查看这个[issue](https://github.com/modelscope/ms-swift/issues/4030)。

### Q26: 想问一下，SWIFT支持蒸馏吗？
参考这个[例子](https://github.com/modelscope/ms-swift/blob/main/examples/sampler/distill/distill.sh)。

### Q27: gkd训练student model和teacher model的model_type需要一致吗，一个dense一个moe可以吗?
可以的，只需要词表一样，不过带MoE就会比较慢。

### Q28: GRPO训练相关问题
SWIFT现在支持多模态的GRPO训练。GRPO训练过程中loss接近0是正常情况，参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)。
设置sleep_mode，推理结束VllmEngine释放显存。下次调用时，再加载，而不是一直占用。
GRPO训练时不想引入KL项，可以通过命令行参数beta设置。
LoRA微调后继续做GRPO训练，请在命令行参数文档中搜索`--adapters`。
由于算entropy会有额外的一点开销，所以默认没有记录熵曲线。如果需要，请设置`--log_entropy true`，
colocate模式不支持use_async_engine。
GRPO不支持channel_loss。
Liger kernel和padding free没法在GRPO阶段一起开。如果一起开，需要改liger grpo loss的实现，在liger kernel库中，不方便改。
如果训练集有不同的task，请查看[多任务训练](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/multi_task.html)。

### Q29: reward函数（模型）相关问题
reward_model和reward_funcs可以一起使用。
自定义reward函数参考[examples/train/grpo/plugin](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin)。
针对math问题，要从数据集里面传solution，不然不好算accuracy。
如果在ORM的自定义奖励函数中需要传入数据集中的某个列，请将该列放到messages之外的其他列。
在GRPO训练的过程中如果需要指定一个llm-judge模型来做打分，请参考奖励模型的文档。

### Q30: rollout相关问题
Rollout应该是不兼容pipeline parallel。
vLLM推理引擎默认trust_rwmote_code为true。

### Q31: 请教一个问题，grpo脚本中的save_steps指的是step还是global step？目前本地训练显示的global step是18， wandb上显示的step是628。
`global_step`，本地tqdm显示的。

### Q32: 默认只用 num_iterations=1 的话，clip 就失去作用了吧？dapo 的 clip higher 也没用。我看 veRL 有个 micro batch 可以设置单轮小批次更新 policy model 来使得 clip 项生效，ms-swift 的 mini batch 看源码貌似只是做了梯度累加？
是的，需要num_iterations>1。

### Q33: 请问gspo训练支持传入参数top_entropy_quantile吗？传入了--importance_sampling_level sequence后，还能实现对熵分布前x%的token的优化吗？
支持，顺序是先正常计算loss（受importance_sampling_level影响），再根据top_entropy_quantile mask掉loss。

### Q34: GRPO文档中的faq
更多GRPO相关的FAQ，请查看[GRPO文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/GetStarted/GRPO.html#faq)

### Q35: ppo等偏好训练相关问题
PPO训练不支持梯度裁剪。
目前PPO还只支持RM和policy是同一系列的模型(tokenizer/template)。
不支持多轮的DPO。

### Q36: MoE模型训练相关问题
MoE模型LoRA训练，如果aux-loss基本没变化，将all-router也加到target_modules。
LoRA训练中，路由器模块是否参与训练看gate是否是nn.Linear实现，如果是nn.Parameter就不训练，详见命令行参数[target_parameters](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#tuner)。

### Q37: Megatron-SWIFT训练相关问题
Checkpoint保存，参考命令行参数[save_strategy](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)。
Megatron多机训练时，因为pp并行只有在pp last rank有完整的信息, 日志在last rank打印，而不是从master node打印。
Megatron-SWIFT支持了save_total_limit，支持了SwanLab监控训练，详见[Megatron-SWIFT命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)
ViT用的是transformers的模型结构，目前没有并行，训练遇到OOM时降低`decoder_first_pipeline_num_layers`。
Megatron-SWIFT支持新的模型，目前没有教程，请查看新增模型的PR。
sequence_parallel的并行数等于tp数。
FP8训练支持block wise，参考[examples/megatron/fp8例子](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/fp8)。

### Q38: 请问Megatron-SWIFT如何配置断点续训？
配置`--mcore_model`加载checkpoint，另外根据需要配置这几个参数，`--finetune`，`--no_load_optim`，`--no_load_rng`。如果是LoRA断点续训，配置`--mcore_adapter`，其他同全参数训练，详见[Megatron-SWIFT命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)。

### Q39: mtp相关问题
需要MTP训练，请设置命令行参数`mtp_num_layers`。
如果base模型不附带MTP结构，可以从头初始化训练MTP。
多模态的MTP目前还没支持。

### Q40: 有个关于Megatron GKD的问题请教，如果teacher是Qwen3-235B，student是Qwen3-30BA3B，之前SFT 235B都是pp8然后decoder fist和decoder last设为11。如果我在GKD的时候也设置decoder first last，会不会影响student的并行？
现在两个模型的并行参数是共用的，不同并行的设置会在v4版本后支持。

### Q41: 量化模型训练相关问题
QLoRA微调参考[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora)。
量化模型不能全参数微调，GPTQ模型的int型参数无法参与求导，只能附着LoRA等额外结构参与更新。
QLoRA训练后的模型merge参考[QLoRA例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora)。
Megatron-SWIFT不支持QLoRA训练。

### Q42: 一些特殊模型的训练
SWIFT目前不支持MiniCPM-O使用音频模态输入的训练。
微调DeepSeek-VL-2，transformers用4.42以前的版本，`peft==0.11.*`。
Moonlight-16B-A3B-Instruct微调。因为模型文件中禁止了训练, 参考DeepSeek-VL-2的解决方案，issue中搜索。
微调Ovis2这个模型有点特殊，需要padding到max_length。设置一下`--max_length`。
Qwen2.5-Omni目前不支持talker训练，只有thinker。
Qwen2-Audio的sft不支持packing。

### Q43: 请问在不支持flash attention的设备上attention implemation默认是什么呢？文档中默认是none
默认使用sdpa。

### Q44: 请问默认模型训练都是left padding是吧?
训练可以选择使用左padding还是右padding。默认是右padding, `batch infer`都是左padding。

### Q45: 请问下MoE的参数有哪些，参数表里关键字搜索不到？专家数量，专家路由这些参数怎么设置？
直接用config.json中的参数。

### Q46: SWIFT能够支持设置最小的learning rate吗，感觉最后减到太小了
可以设置，`--lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'`。

### Q47: 目前支持用yaml文件配置grpo和sft吗？
都支持的，该配置是在main.py中直接处理成命令行。

### Q48: 请问现在是不支持use_liger_kernel和log_entropy一起用吗？
不支持。

### Q49: 请问下，遇到这个报错，怎么处理？安装了apex也不行
```text
RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found. To use gradient_accumulation_fusion you must install APEX with --cpp_ext and --cuda_ext. For example: pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
```
设置一下`--gradient_accumulation_fusion false`。

### Q50: 几个任务一起finetune vlm，不同任务视频采样规则不一致，ms-swift是否支持？在哪里配置？
[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)看下`interleave_prob`。

### Q51: 想问一个问题，多模态packing预训练每次pytorch allocator cache flushes since last step后，显存使用好像就会增长一点，步数多了容易oom
加个环境变量`PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`。

### Q52: use_logits_to_keep 现在多模态大模型上可以用吗？
如果多模态token的展开在模型的forward内会报错。

### Q53: 请问一下为什么训练到会有好几次显存大幅度增加，已经50step或者100step
设置环境变量`PYTORCH_CUDA_ALLOC_CONF`，具体查看PyTorch文档。

### Q54: 从qwen base模型微调成chat模型有没有实践文档，有什么要特别配置的吗?
`swift sft`，没有其他需要特别配置的，参考[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat)。

### Q55: 模型训练后，回复重复了很多内容
参考[预训练与微调](https://swift.readthedocs.io/zh-cn/latest/Instruction/Pre-training-and-Fine-tuning.html)。如果训练过程中出现重复的情况，请多训练几个epoch, 清洗数据, 全参数训练, 采用RLHF的方式缓解。

### Q56: 请问为什么 --torch_dtype float16 （卡不能使用bf16）会出现报错：lib/python3.12/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_ raise ValueError("Attempting to unscale FP16 gradients.") ValueError: Attempting to unscale FP16 gradients.
全参数，不能fp16训练的。

### Q57: 请问下，lora参数合并报错，目前peft是0.11.0，这个是因为peft版本需要升级吗
```text
File "/opt/conda/lib/python3.9/site-packages/peft/config.py", line 118, in from_peft_type
  return config_cls(**kwargs)
TypeError: __init__() got an unexpected keyword argument 'corda_config'
```
训练和合并的peft版本不一致导致的。

### Q58: 请问这个问题如何解决？safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
磁盘空间不足了，模型没有保存完整。

### Q59: 这个错误为什么会出现在这，numpy.object找不到在哪？
`numpy==1.26.3`，尝试一下。

### Q60: unsloth训练，报错：assert(type(target modules) in (list,tuple,))。配置的参数是--target modules all-linear
别用`all-linear`，改为具体的模块列表，比如`--target_modules q k v`。

### Q61: 请问对于qwen2.5-omni来说--freeze_vit false意味这视觉编码器和音频编码器都打开了，有什么办法可以只打开音频编码器不打开视觉编码器吗？
`--target_regex`写一下。

## 推理

SWIFT支持python脚本、命令行、ui界面推理，详见[推理和部署](https://swift.readthedocs.io/zh-cn/latest/Instruction/Inference-and-deployment.html)。

### Q1:SWIFT推理如何设置模型？
如果是全参数训练的模型、LoRA训练后合并的模型或者从model hub下载的模型，设置命令行参数`--model <model_id_or_path>`；LoRA训练后未合并的模型，用`--adapters`设置，同时可通过`--model`指定基模路径。

### Q2: SWIFT如何使用数据集进行推理？推理结果保存在哪儿？
`--val_dataset <your-val-dataset>`，指定数据集。对于训练后的模型也可以设置参数`--load_data_args true`。推理结果保存路径通过`--result_path your_path`设置，日志中会打印路径。详见文档[命令行参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。
如果需要保留推理数据集中额外的字段，请设置`--remove_unused_columns false`。

### Q3: SWIFT如何设置批量推理？
如果infer_backend为`transformers`，设置命令行参数`--max_batch_size 16`，或[python脚本](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py)。这里max_batch_size指的是每张卡上的batch_size。

### Q4: SWIFT如何设置流式推理？
`--stream true`，此时推理结果将逐条写入jsonl文件。需要注意的是，流式推理不支持ddp。

### Q5: vLLM和SGLang推理后端相关的问题
对于LoRA训练的模型，请查看vLLM和SGLang文档，如果支持LoRA推理则不需要合并。此外，SGLang推理目前不支持多模态。

### Q6: 生成参数相关的问题
temperature等参数默认从generation_config.json中读取。设置`--temperature 0`或者`--top_k 1`可以取消推理随机性。

### Q7: 如何将system_prompt置空？命令行不设置system参数，但是它会加上默认的system。
设置`--system ''`。

### Q8: 推理时如何计算acc/rouge等指标？
参考[推理参数metric](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id14)。

### Q9: 模型推理的时候如果需要在特定前缀下继续推理的话是设置哪个参数？
参数`--response_prefix`。

### Q10: 数据answer里面已经包含了部分prompt，希望补全answer，应该怎么修改inference？
```text
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "answer1, "}]}
```
用swift3.0以后的版本是可以的，参考[examples/infer/demo_agent](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py)。

### Q11: 多模态模型推理时如何限制最大像素，以减少显存占用？
设置命令行参数`--max_pixels xxx`、环境变量`MAX_PIXELS=xxx`、或特定模型参数`--model_kwargs '{"max_pixels": xxx}'`，其中环境变量仅对文档中对应的模型生效，详见文档[特定模型参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id18)。

### Q12: SWIFT推理如何输出概率值logprobs参数？
命令行推理设置`--logprobs true`，python脚本推理设置`request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`，参考[test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/infer/test_logprobs.py)。

### Q13: SWIFT推理如何输出last_hidden_state？
没有例子，可以参考GRPO trainer的`_get_last_hidden_state`方法。

### Q14: transformers，vllm，ollama等推理结果不一致问题
SWIFT的template是对齐transformers的。检查推理参数是否对其。此外，VllmEngine和TransformersEngine是有差异的。

### Q15: embedding/reranker模型推理
embedding模型推理参考这里的[例子](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_embedding.py)。reranker模型推理参考这里的[例子](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_reranker.py)。

### Q16: 请问在使用python脚本推理时，如何使用cpu?
设置环境变量，`os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`。

### Q17: 使用swift infer命令进行推理，支持多机推理吗？
如果单节点放得下模型，外面封装k8s就行。如果单节点放不下那就不支持。

### Q18: swift sample的时候，好像不支持batch？好像是for循环一个个例子sample，有点慢
有一个[脚本](https://github.com/modelscope/ms-swift/blob/main/examples/train/rft/rft.py)，可以用多进程对数据集拆分采样。

### Q19: 特殊模型依赖版本相关问题
Qwen2-Audio推理结果出现混乱，请使用transformers4.48。
transformers4.55.2训练的LoRA不能使用小于4.52的版本加载了，详见[issue#5440](https://github.com/modelscope/ms-swift/issues/5440)。
swift对不同版本的qwen-vl-utils做了兼容，使用qwen2.5-vl和qwen3-vl模型时不需要切换该依赖版本。

### Q20: 报错，safetensors_rust.SafetensorError: Error while deserializing header:MetadataIncompleteBuffer
模型权重损坏了。

## 导出

### Q1: autoawq相关的报错
如果推理没有涉及AWQ量化模型，但出现了autoawq相关的报错，可以尝试卸载autoawq再进行推理。不支持AWQ量化的模型，尝试用GPTQ进行量化。

### Q2: SWIFT量化模型时，一张卡上放不下模型的情况
尝试设置`--device_map cpu`。或者多卡加载模型，单卡量化。

### Q3: 想问一下用swift export对qwen2.5 72B模型进行gptq int4量化，max model length=32768用的是默认值，给的校准数据集有128个样本，但是量化的时候报错了，报错日志是：factorization could not be completed because the input is not positive-definite(the leading minor of order 18145 is not pisitive-definite)。是什么原因？
海森矩阵不正定的问题，试试其他的数据集。

### Q4: swift export的时候传入自定义的template_type,是不是就可以永久改掉template_type了？如果swift export --template_type 自定义,是不是就可以把模型对应的template改掉
不会被修改,swift中的template是定义在swift内部的,不是以jinja方式保存的。

### Q5: 模型训练完能直接转gguf格式吗？
目前只支持导出ModelFile，详见文档[命令行参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。

## 部署

### Q1: SWIFT部署如何设置模型？
同上面推理Q1。

### Q2: SWIFT如何进行多卡部署？
详见[例子](https://github.com/modelscope/ms-swift/tree/main/examples/deploy)。如果是transformers engine，不支持DDP，不能多卡部署。此外，不支持异构部署，如不同型号的显卡、各显卡设置不同的存储占比等。

### Q3: 通过--system参数指定system prompt与数据集中每个数据前加system prompt以及template的system prompt是不是有一个就行？这些方式对模型来说，是不是一样的？
system优先级：数据集中的>命令行的>template中默认的。

### Q4: 客户端多模态输入相关问题
客户端传入图片、音频等，见[客户端例子](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm)。
如果图片url非法，可以设置请求的超时时间，环境变量`SWIFT_TIMEOUT`，或者`InferClient`中可以传参数。

### Q5: 生成参数设置相关问题
temperature等参数推理只能启动前设置，部署可以在启动时设置默认值，之后在客户端继续设置，覆盖默认值。

### Q6: SWIFT部署的模型怎么设置流式生成？
客户端控制的，查看[examples/deploy/client](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client)。

### Q7: SWIFT部署如何输出token的概率？
服务端设置`--logprobs true`，要客户端传参数，`request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`。

### Q8: 部署模型时，thinking相关问题
如果需要禁止思考，目前只能在swift deploy启动的时候禁止thinking。查看这个[issue](https://github.com/modelscope/ms-swift/issues/4030)。

### Q9: 部署时，设置什么参数可以实现一次输出多个结果？
`RequestConfig`参数`n`。

### Q10: SWIFT部署，指定--infer_backend vllm，和直接使用vllm部署相关问题
如果两者推理结果相差较多，可能是template没对齐。如果推理速度相差较多，可能是图像分辨率不一致。swift默认使用V1 engine，可以通过环境变量`VLLM_USE_V1=1`控制。

### Q11: 特殊模型和依赖版本相关问题
如果遇到报错没有“model.language_model.embed_tokens.weight”，训练前后的transformers版本不一致。
qwen2.5使用fp16推理如果遇到返回乱码，尝试bf16。

### Q12: 有个问题想问一下，qwen2-7b部署后使用客户端时，调用openai的api要使用client.completions.create，不能使用client.chat.completions.create，但是使用qwen2-7b-instruct-q5_k_m.gguf的时候可以使用client.chat.completions.create，这是为什么呀？
base模型可以用client.chat.completions.create的，不过这个是兼容行为。

## 评测

### Q1: SWIFT支持的评测集有哪些？以及如何使用自定义评测集？
标准评测集和用户自定义评测集的使用详见文档[评测](https://swift.readthedocs.io/zh-cn/latest/Instruction/Evaluation.html)。

### Q2: 官方支持的评测数据集手动下载后，swift eval能配置本地路径评测吗？
离线评测请参考EvalScope文档[快速上手](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)

### Q3: eval微调后的模型，总是会在固定的百分比停掉，但是vllm服务看着一直是有在正常运行的。模型越大，断开的越早。
`SWIFT_TIMEOUT`环境变量设置为-1。

### Q4: 评估的时候可不可以控制数据集条数？评估一个mmlu需要一个多小时，也太慢了。
配置参数`--eval_limit`，这里的`--eval_limit`是控制了每个subset的条数，比如mmlu有50多个subset，每个limit10条，那就是500多条。

### Q5: 问一下评估swift eval里，模型最多生成1024token就结束了，这个如何修改？设置--max_new_tokens 5000，看起来没起作用
查看命令行参数[eval_generation_config](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id16)

### Q6: 请教一下，想使用OpenCompass的后端评测，如何从本地加载下载好的数据集？
OpenCompass后端不支持设置`data_args`。

### Q7: swift eval 来评估模型，--eval_backend OpenCompass不支持自定义数据集吗？
```text
ValueError: eval_dataset: /mnt/workspace/data.jsonl is not supported.
eval_backend: OpenCompass supported datasets: ['C3', 'summedits', 'WiC', 'csl', 'lambada', 'mbpp', 'hellaswag', 'ARC_e', 'math', 'nq', 'race', 'MultiRC', 'cmb', 'ceval', 'GaokaoBench', 'mmlu', 'winogrande', 'tnews', 'triviaqa', 'CB', 'cluewsc', 'humaneval', 'AX_g', 'DRCD', 'RTE', 'ocnli_fc', 'gsm8k', 'obqa', 'ReCoRD', 'Xsum', 'ocnli', 'WSC', 'siqa', 'agieval', 'piqa', 'cmnli', 'cmmlu', 'eprstmt', 'storycloze', 'AX_b', 'afqmc', 'strategyqa', 'bustm', 'BoolQ', 'COPA', 'ARC_c', 'PMMEval', 'chid', 'CMRC', 'lcsts']
```
OpenCompass不支持自定义数据集，用native可以自定义模式。

### Q8: evalscope原生是可以生成报告的，其他后端如opencompass是不支持生成报告可视化是吗？
目前只支持native的可视化，其他后端还不支持。

### Q9: 请问一下评测ifeval报这个错是什么原因？
```text
[Errno 20] Not a directory: '/root/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'
```
解压这个文件，`unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`。

### Q10: 请问评测时eval_backend='OpenCompass'，怎么指定离线数据集路径？
查看[数据准备教程](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html#id3)，下载数据集并解压。不用指定`dataset-args`，将数据集文件夹（即data文件夹）放置在当前工作路径下即可。

### Q11: 用evalscope报这个错是什么原因？
```text
unzip: cannot find or open /root/nltk_data/tokenizers/punkt_tab.zip, /root/nltk_data/tokenizers/punkt_tab.zip.zip or /root/nltk_data/tokenizers/punkt_tab.zip.ZIP
```
这是在下载nltk的依赖，手动下载[punkt_tab.zip](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip)，解压到`~/nltk_data/tokenizers`下面。

### Q12: 为啥纯文本没问题，测多模态我们指定路径了，但他还是检测不到数据集，会去下载？
VLMEvalKit流程跟native不一样，会自己下载数据放到`~/LMUData/`下面，详见[文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#id2)。

### Q13: 请问一下swift eval做benchmark评测的时候，是否可以指定llm作为judge, 参数应该怎么传进去？
支持，使用swift得从`extra_eval_args`去传递`judge-model-args`参数，包括`api_key，api_url，model_id`，整体是一个json字符串。

### Q14: 请问在执行eval的时候出现了多卡显存分配不均是什么原因？
```shell
NPROC_PER_NODE=8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\ MAX_PIXELS=802816\ swift eval\
--model "$MODEL_PATH” \$EXTRA_ARGS \
--eval_backend Native \ --infer_backend transformers\ --device_map auto \
--eval_limit"$EVAL_LIMIT"\ --eval_dataset general_qa\
--dataset_args "{\"general_qa\": {\"local_path\": \"${DATA_PATH}\", \"subset_list\": [\"${SUBSET_NAME}\"]}}" \ --host 127.0.0.1\> "$LOG_FILE" 2>&1
```
swift eval不支持DDP方式启动。

### Q15: 请问哪里可以看到swift评测的时候送入的query除了问题之外还有哪些额外的字段呢？
最简单的方法是看输出的reviews文件中的input字段，是输入给模型的内容转换后的Markdown格式。如果用backend是opencompass的话没有这些，需要用native backend。

ms-swift的eval能力使用了魔搭社区评测框架EvalScope, 复杂能力请直接使用[EvalScope框架](https://evalscope.readthedocs.io/zh-cn/latest/get_started/introduction.html)。
