# 常见问题整理

下面是SWIFT使用过程中遇到的一些常见问题。

## 数据集

SWIFT内置150+数据集，可用于预训练、微调、人眼对齐、多模态等各种任务，并支持自定义数据集。详见[主页](https://github.com/modelscope/ms-swift/blob/main/README_CN.md)。

### Q1: SWIFT支持的数据集有哪些？如何使用自定义数据集？如何下载数据集？如何检查数据集？
- 支持的数据集详见[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)。
- 自定义数据集格式及使用方法详见[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)，符合格式的数据集会自动调用swift内置数据预处理器。如果与文档要求格式不一致，请参考已支持的数据集自行转换格式。若自定义数据集中有额外的字段，这些字段默认不会被使用，可以通过`--remove_unused_columns`进行额外的设置。
- 需要下载数据集，然后通过路径指定方式使用时，可以通过`git clone`下载到本地，通过 dataset_info.json 文件中的`dataset_path`字段指定。具体请查看[自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#dataset-info-json)。数据集的下载模式可以选择重新下载或重用上次下载，通过`--download_mode`指定。
- 对数据集进行错误检查，请设置命令行参数`--strict True`。需要数据集质检工具时，可以查看另一个库[data-juicer](https://github.com/modelscope/data-juicer)。数据随机可以使用`--dataset_shuffle true`。
更多说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索对应参数。

### Q2: 常见数据集报错
- 由于datasets底层pyarrow对于类型管控较为严格，图像grounding数据集的objects部分、agent数据集的tools部分等，需要使用str类型，否则会报错每行的类型不一致。
- 训练中如果遇到报错`AttributeError:’TrainerState’ object has no attribute ’last_model_checkpoint’`，可能是因为数据集过少，导致数据数量不足一个step，可以尝试扩充数据集解决。同理，切分的验证集数据很少时也会有类似报错。
- 下面是一个assistant字段为空导致的报错：
```shell
File "/your_workspace/ms-swift/swift/1lm/dataset/preprocessor/core. py", line 69, in _check_messages raise
ValueError(f'assistant_message; {assistant_message}')
ValueError: assistant_message: {'role' :'assistant', 'content': ''}
```
如果是推理使用，可以直接删掉空assistant message。

### Q3: 从缓存加载数据集相关问题
设置命令行参数`--load_from_cache_file True`，可以加快数据集的加载速度（非初次加载），尤其是在多模态数据集、数据集本身数据量较大等场景。在debug或修改preprocessor时，设置为false，以确保代码改动生效。更多说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索该参数。

### Q4: 多模态模型数据集相关问题
- 多模态模型训练的[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal)。支持纯文本或图文数据训练，也支持两种数据混合进行训练。图像、视频、音频相关的参数，如，最大像素、fps等请查看[特定模型参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id19)。
- Grounding任务中通用数据格式支持了一个物体对应多个bbox，参考文档[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#grounding)。训练时SWIFT会对超过`--max_pixels`的图像进行调整，并保存预处理前和后的图像，同时对bbox进行调整，推理阶段不会进行调整，需要提前手动处理图像。

### Q5: 大规模数据集相关问题
数据集太大了，然后每次tokenize都需要很久，请使用`--lazy_tokenize True`或流式读取`--streaming True`，详见[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。

### Q6: 数据集流式加载相关问题
流式加载`--streaming True`，一边训练一边加载，需要设置max_steps，详细说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id4)中搜索该参数。<br>
注意：
- streaming是不随机的，也不划分验证集，验证集通过命令行参数`val_dataset`指定。
- 断点续训时，流式只能往前索引，不能随机索引，跳过已经训练的数据耗时特别长，不建议用流式。

### Q7: 数据集多进程处理
多模态数据集map比较慢是正常的，可以设置参数`--dataset_num_proc`开多进程提速。

## 训练

SWIFT支持的训练方法包括预训练、指令监督微调、偏好学习、GRPO、Embedding、Reranker、序列分类任务等，详见[主页](https://github.com/modelscope/ms-swift/blob/main/README_CN.md)。

### Q1: 如何搭建SWIFT环境？如何离线安装SWIFT？有镜像可以使用吗？
- 环境搭建详见[SWIFT安装文档](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)，一些常见依赖的推荐版本可以在[GitHub主页](https://github.com/modelscope/ms-swift/blob/main/README_CN.md)上找到。
- SWIFT离线安装流程：
```text
1、git clone下来（需要联网）
2、在本地pip install -e .
```
- [SWIFT安装文档](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)中提供了镜像地址，用`docker pull`拉取、`docker run`命令启动容器即可，如：
```shell
# 拉取镜像
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1
# 后台拉起容器，-d会使容器在后台长期运行
docker run --gpus all -p 8000:8000 -dit --name ms modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1 /bin/bash
# 进入容器
docker exec -it ms /bin/bash
```
启动容器后拉SWIFT最新代码安装即可。

### Q2: SWIFT支持的模型有哪些？如何下载模型？如何设置模型存储路径？
- 支持的模型详见[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)。
- 如果模型已经下载到了本地，可以通过设置`--model <model_path>`使用。对于离线环境训练，需要同时设置`--model <local_path_to_model>`和`--check_model false`。如果提示git clone相关报错，可以通过`--local_repo_path <local_repo_path>`指定本地repo。
- 从ModelScope下载的模型，可以通过配置环境变量`MODELSCOPE_CACHE=your_path`将模型存到指定路径。如果用ModelScope SDK下载，同样可以通过`--cache_dir="local_path"`来指定模型存放路径。模型下载还可以使用`modelscope download`命令行工具或`git`下载，详见modelscope文档中的[模型下载](https://modelscope.cn/docs/models/download)。如果需要从Hugging Face下载模型，需要设置环境变量`USE_HF=1`。
- SWIFT会自动匹配model_type，也可以查看[支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)，手动指定。
更多说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索对应参数。

### Q3: template相关问题
- 由于jinja chat template没有labels，所以目前不支持该template的训练。
- 多模态数据集如果需要在加载数据之后做动态数据增强（例如，给输入数据随机添加噪声等），请在template中修改`encode`方法。

### Q4: SWIFT训练如何debug？
可以使用以下方式进行debug，这与使用命令行微调是等价的，但此方式不支持分布式。微调命令行运行入口可以查看[这里](https://github.com/modelscope/ms-swift/blob/main/swift/cli/sft.py)。
```shell
from swift import sft_main, SftArguments
result = sft_main(SftArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    tuner_type='lora',
    dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500',
             'AI-ModelScope/alpaca-gpt4-data-en#500',
             'swift/self-cognition#500'],
    torch_dtype='bfloat16',
    # ...
))
```

### Q5: SWIFT如何使用python脚本训练？
参考[notebook示例](https://github.com/modelscope/ms-swift/tree/main/examples/notebook)。

### Q6: SWIFT如何使用UI界面训练？
- 使用`swift web-ui`命令启动UI界面，界面训练、自定义数据集使用与命令行一致，界面上的参数请查看[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。
- Megatron-SWIFT不支持UI界面训练。

### Q7: 多模态模型训练相关问题
- VLM模型训练如果需要减少显存使用，请配置`--freeze_vit true`，以及限制最大像素`--max_pixels`。如果没有训练VIT，抛出`warning: none of the inputs have requires_grad=True`是正常的，如果训练了，则不应该抛出。
- `--freeze_vit`，`--freeze_aligner`，`--freeze_llm`这几个参数详见[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#tuner)。
- 全参数微调visual encoder+LoRA微调LLM，参考[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit)。

### Q8: 单机多卡训练相关问题
SWIFT多卡训练底层依赖torchrun。`deepspeed` 和 `device_map`不兼容，两个只能选1个。更多细节请查看代码库中的[单机多卡例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu)。

### Q9: 多机多卡训练相关问题
- 多机多卡训练时，只有主节点有日志。更多细节请查看代码库中的[多机多卡例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)。
- 多机训练速度缓慢，如，使用DeepSpeed ZeRO3训练会出现严重的速度下降，请查看[issue](https://github.com/modelscope/ms-swift/issues/1825)。

### Q10: 断点续训相关问题
- 先前训练脚本中的参数不变，加上`--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`即可，权重等相关信息将在trainer中读取。
- 如果希望仅加载模型，请同时设置`--resume_only_model`来忽略优化器状态和随机种子。
- 更复杂的场景，请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索参数关键词`resume`。

### Q11: packing相关问题
- packing要和flash_attn一起使用，不然attention_mask会出问题，导致误差。
- Qwen3.5模型中的linear-attention不支持var_len，不建议开启packing。
- 开启packing时，多模态数据会有两次mapping，map完数据集之后还会进行template的mapping。如果速度非常慢，可以设置`OMP_NUM_THREADS=14`加速，或者可以把packing去掉，就不会map第二次了。

### Q12: 当前训练完默认保存多少个checkpoint？
默认保存所有的checkpoint，详见[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中的save_total_limit。

### Q13: 训练过程中loss相关问题
- 自定义损失函数的.py文件在使用时通过`--external_plugins`参数导入：
```shell
swift sft \
    --external_plugins /path/to/plugin.py \
    --loss_type my_loss \
    # ...
```
其中`--loss_type`对应的值为loss_map中注册的自定义损失函数对应key；lose_scale同理，需要注册在loss_scale_map中。
- 如果需要不同数据集的loss曲线，请设置`--enable_channel_loss`。更多说明请在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索该参数。
- 可以在[loss_map](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py)中查看当前支持的loss或添加新的loss
- 如需检查`<image>`等特殊token是否参与损失计算，可以在命令行日志中找打印出的labels对应检查。
- 训练agent时，`tool_call`算loss，`tool_response`不算loss。

### Q14: 训练过程中acc相关问题
- 如果eval得到的acc和对应ckpt重新推理一遍计算得到的acc不一致，可能是因为训练时候的eval_acc和推理时候的acc计算方式不一样导致的。检查一下`--acc_strategy`参数，默认为`'token'`, 可选择的值包括: `'token'`，`'seq'`。
- 有些模型训练过程中没有token_acc是因为`logits`和`labels`数量对不上。

### Q15: 模型参数freeze相关问题
- DDP多卡训练的过程中，冻结某些层时导致某些参数未参与梯度回传，请配置`--ddp_find_unused_parameters True`自动跳过没有梯度的参数。
- `--freeze_parameters/--freeze_vit/--freeze_aligner/--freeze_llm`：这四个参数设置的freeze_parameters在使用的过程中允许被后执行的activate_parameters覆盖，即参数解冻优先级更高。
- `--freeze vit/--freeze aligner/--freeze llm`这三个参数会对freeze parameters进行调整。因为有些模型的ViT中包含aligner，`--freeze aligner False`时会同步调节trainable parameters，将`aligner`单独加入其中确保不被冻结。
- `--freeze_parameters_ratio`这个参数的机制是从embedding开始从下往上冻结参数。

### Q16: 序列并行相关问题
- pt，sft，dpo，grpo均支持sequence parallel。命令行例子请参考[sequence_parallel](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel)。
- VLM模型目前仅支持flash-attn，纯文本支持flash-attn和sdpa。
- sequence parallel可以和Liger kernel同时使用。
- sequence parallel下自定义loss不起作用时，可能是由于sequence parallel走了自己的loss，可以根据情况修改[per_token_loss_func_sp](https://github.com/modelscope/ms-swift/blob/main/swift/trainers/utils.py)。

### Q17: 扩充词表
用SWIFT框架扩充词表需要设置命令行参数`--new_special_tokens <path/to/tokens.txt>`配合`--modules_to_save embed_tokens lm_head`来解冻对应参数进行训练，详见[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens)。

### Q18: tuners相关问题
- SWIFT中的LlamaPro对多模态做了适配。
- LongLoRA因为依赖架构中的特定组件，所以只有LLaMA系列模型能用。
- LoRA训练和`--trainable_parameters`参数不兼容，需要额外训练LoRA模块之外的其他参数用`--modules_to_save`。

### Q19: embedding/reranker训练相关问题
- [embedding训练例子](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding)。
- [reranker训练例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/reranker)。
- embedding/reranker数据格式见[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)。

### Q20: classification训练相关问题
- 需设置`--num_labels`、`--problem_type`。详细介绍在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索对应参数。
- 多标签分类数据格式见[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)。<br>
注意：数据集中label字段和message字段同级。

### Q21: thinking模型训练
查看这个[issue](https://github.com/modelscope/ms-swift/issues/4030)。

### Q22: SWIFT支持蒸馏吗？
参考这个[例子](https://github.com/modelscope/ms-swift/blob/main/examples/sampler/distill/distill.sh)。

### Q23: GKD训练相关问题
- GKD训练支持student model和teacher model的model_type不一致，只需要词表一样即可（带MoE会比较慢）。
- SWIFT v4版本以后支持teacher model和student model实行不同的并行配置。详情请见[例子](https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd)。

### Q24: GRPO训练相关问题
- SWIFT现在支持多模态的GRPO训练。GRPO训练过程中loss接近0是正常情况，参考[issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851)。
- GRPO训练时不想引入KL项，可以通过设置KL正则系数`--beta=0`不加载ref model。
- LoRA微调后继续做GRPO训练，使用`--adapters sft_ckpt --ref_adapters sft_ckpt`。
- 由于算entropy会有额外的一点开销，所以默认没有记录曲线。如果需要，请设置`--log_entropy True`，
- colocate模式不支持`--vllm_use_async_engine`。
- GRPO不支持channel_loss。
- GRPO无法同时使用Liger kernel和padding free。同时使用需要修改liger kernel库中的liger grpo loss。
- GRPO/PPO代码实现中mini_batch仅用于梯度累积。激活Clip机制需要num_iterations>1。设置num_iterations=1时会导致失效。
- 如果训练集有不同的task，请查看[多任务训练文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/multi_task.html)。
- 更多GRPO相关的FAQ，请查看[GRPO FAQ](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/GetStarted/GRPO.html#faq)

### Q25: Reward函数（模型）相关问题
- `--reward_model`和`--reward_funcs`可以一起使用，最终会通过加权求和得到一个总reward。加权权重可以通过`--reward_weights`指定，权重顺序为reward_func1、reward_func2、..、reward_funcn、reward_model。
- 自定义reward函数参考[examples/train/grpo/plugin/plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py)。
- 针对math问题，数据集里需要存在solution字段，不然影响accuracy的计算。
- 如果在ORM的自定义reward函数中需要传入数据集中的某个字段，请将该字段放到与messages同级的位置，之后可以从reward_kwargs中拿到该字段。
- 在GRPO训练的过程中如果需要指定一个llm-judge模型来做打分，请参考[奖励模型的文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/reward_model.html)来实现。

### Q26: Rollout相关问题
- Rollout不兼容Pipeline Parallel，如果需要多卡推理加速，可以使用Tensor Parallel。
- vLLM推理引擎默认`trust_remote_code`为true。

### Q27: 训练脚本中的save_steps指的是step还是global step？
指的是global_step，也就是本地tqdm显示的。

### Q28: GSPO训练传入了`--importance_sampling_level sequence`后，还支持传入参数`--top_entropy_quantile`吗？即还能实现对熵分布前x%的token的优化吗？
支持，顺序是先正常计算Sequence loss（受importance_sampling_level影响），再根据top_entropy_quantile mask loss。

### Q29: ppo等偏好训练相关问题
- PPO训练不支持`--max_grad_norm`参数，如果出现梯度爆炸，需要从学习率、reward scale等其他方面调参。
- 目前PPO还只支持RM和policy是同一系列的模型(tokenizer/template)，不然会导致prompt格式不一致、token序列切分不一致等问题影响效果。
- 目前不支持多轮的DPO，可以用 GRPO + multi-turn（多轮推理 + reward 函数打分）作为替代。

### Q30: MoE模型训练相关问题
LoRA训练中，路由器模块是否参与训练取决于gate/router是否是nn.Linear实现。如果是nn.Parameter实现不会参与LoRA训练，这种情况会导致aux-loss基本没变化，在此前提下如果希望训练路由器，需要将all-router也加到`--target_modules`。
```shell
--target_modules all-linear all-router
```
all-router不是通配符匹配模块名，而是一个特殊关键字，告诉框架"把路由器也纳入可训练范围"。<br>
还可以通过`--target_parameters`来指定LoRA替换具体参数，详见命令行参数[target_parameters](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#tuner)。

### Q31: Megatron-SWIFT训练相关问题
- Checkpoint保存，搜索命令行参数[--save_strategy](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)。
- Megatron多机训练pipeline parallel时只有last rank持有完整输出，所以日志在last rank打印，而不是从master node打印。
- Megatron-SWIFT支持`--save_total_limit`；支持SwanLab监控训练，详见[Megatron-SWIFT命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)
- ViT用的是transformers的模型结构，目前不走Megatron并行。训练遇到OOM时可以通过`--decoder_first_pipeline_num_layers`参数来降低LLM decoder层数，留更多显存给ViT来缓解。
- Megatron-SWIFT支持新增模型，但是目前没有教程，请查看新增模型的PR了解配置方式。
- Megatron-SWIFT的sequence parallel不是独立设置的，并行度等于tensor parallel的度数，即通过`--tensor_parallel_size`设置。
- 支持Block-wise FP8，参考[examples/megatron/fp8例子](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/fp8)。
- 断点续训需要配置如下参数。
```shell
--mcore_model <path/to/checkpoint-xxx>   # 加载模型权重
--finetune false                         # 标记为微调模式（而非继续训练）
--no_load_optim                          # 不加载 optimizer 状态（可选）
--no_load_rng                            # 不恢复随机数状态（可选）
```
如果是LoRA断点续训，需要额外设置`--mcore_adapter`，其他同全参数训练，详见[Megatron-SWIFT命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Megatron-SWIFT/Command-line-parameters.html)。
- Megatron-SWIFT不支持QLoRA训练。

### Q32: mtp相关问题
- 需要MTP训练，请手动设置命令行参数`--mtp_num_layers`，可以参考config.json中的`num_nextn_predict_layers`，`mtp_num_hidden_layers`字段填写该值。
- 如果base模型不附带MTP结构，可以从头初始化训练MTP。
- 多模态的MTP目前还没支持。

### Q33: 量化模型训练相关问题
- QLoRA微调参考[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora)。
- GPTQ等int类型的量化方式导致参数无法参与求导，所以无法进行全参数微调，只能附着LoRA等额外结构参与更新。
- QLoRA训练后的模型merge参考[QLoRA例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora)。
- Megatron-SWIFT不支持QLoRA训练。

### Q34: 一些特殊模型的训练
- SWIFT目前不支持MiniCPM-O使用音频模态输入的训练。
- 微调DeepSeek-VL-2需要`transformers<4.42`，`peft==0.11.*`。
- Moonlight-16B-A3B-Instruct微调，因为模型文件中禁止了训练, 参考DeepSeek-VL-2的[解决方案](https://github.com/modelscope/ms-swift/issues/543)绕过。
- Ovis2这个模型有点特殊，微调时需要padding到max_length，所以要显式设置`--max_length`。
- Qwen2.5-Omni目前不支持talker训练，只有thinker。
- Qwen2-Audio的sft不支持packing。

### Q35: 在不支持flash attention的设备上attention implemation默认是什么？
默认使用sdpa。

### Q36: 模型的默认训练都是left padding吗?
训练可以选择使用left padding还是right padding。默认是right padding, batch infer都是left padding。

### Q37: SWIFT能够支持设置最小的learning rate吗，感觉最后减到太小了
可以设置，通过
```shell
--lr_scheduler_type cosine_with_min_lr
--lr_scheduler_kwargs '{"min_lr": 1e-6}'
```

### Q38: 是否支持用yaml文件配置grpo和sft？
支持，该配置会在main.py中被处理成命令行。

### Q39: 是否支持`--use_liger_kernel`和`--log_entropy`一起用？
不支持，liger没有实例化logits，无法获取entropies。

### Q40: 遇到gradient_accumulation_fusion相关报错，安装了apex也不无法解决
```shell
RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found. To use gradient_accumulation_fusion you must install APEX with --cpp_ext and --cuda_ext. For example: pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
```
通过`--gradient_accumulation_fusion false`关闭梯度累积融合。

### Q41: 几个任务一起finetune vlm，不同任务视频采样规则不一致，如何配置？
在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)中搜索`--interleave_prob`。

### Q42: 多模态packing预训练每次pytorch allocator cache flushes since last step后，显存使用好像就会增长一点，步数多了容易oom
添加环境变量`PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`，减少内存碎片化。

### Q43: `--use_logits_to_keep`在多模态大模型上可以用吗？
如果多模态的token展开发生在模型外部，则可用；如果发生在模型forward内部，就报错。

### Q44: 从qwen base模型微调成chat模型有没有实践文档，有什么要特别配置的吗?
使用`swift sft`即可，没有其他需要特别配置的，参考[例子](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat)。

### Q45: 模型训练后，回复重复了很多内容怎么办？
请参考[预训练与微调](https://swift.readthedocs.io/zh-cn/latest/Instruction/Pre-training-and-Fine-tuning.html)。如果训练过程中出现重复的情况，可以考虑多训练几个epoch, 清洗数据, 全参数训练或者采用RLHF的方式来缓解。

### Q46: 全参数训练的时候由于卡不能使用bf16，所以设置`--torch_dtype float16`，出现以下报错
```shell
lib/python3.12/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_ raise ValueError("Attempting to unscale FP16 gradients.") ValueError: Attempting to unscale FP16 gradients.
```
fp16 的数值范围很小（最大 65504），全参数训练的梯度容易溢出。可以使用`--torch_dtype fp32`尝试一下。

### Q47: lora参数合并出现以下报错，目前peft是0.11.0，这个是因为peft版本需要升级吗？
```shell
File "/opt/conda/lib/python3.9/site-packages/peft/config.py", line 118, in from_peft_type
  return config_cls(**kwargs)
TypeError: __init__() got an unexpected keyword argument 'corda_config'
```
训练端和合并端的peft版本不一致导致的，合并端需要升级peft到和训练端一致（或更高）的版本。

### Q48: safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
磁盘空间不足了，模型没有保存完整，权重数据被截断。

### Q49: AttributeError: module 'numpy' has no attribute 'object'
`numpy==1.26.3`，尝试一下。

### Q50: unsloth训练，报错：assert(type(target modules) in (list,tuple,))。配置的参数是`--target modules all-linear`
将`all-linear`改为具体的模块列表，比如`--target_modules q k v`，unsloth的LoRA实现路径不会展开具体模块名。

### Q51: 对于qwen2.5-omni来说--freeze_vit false意味这视觉编码器和音频编码器都打开了，有什么办法可以只打开音频编码器不打开视觉编码器吗？
用`--target_regex`正则匹配只想要训练的模块路径。例如：
```shell
--target_regex ".*audio.*"   # 只匹配包含 audio 的模块
```

## 推理

SWIFT支持python脚本、命令行、ui界面推理，详见[推理和部署](https://swift.readthedocs.io/zh-cn/latest/Instruction/Inference-and-deployment.html)。

### Q1:SWIFT推理如何设置模型？
- 全参数训练的模型、LoRA训练后合并的模型或者从model hub下载的模型，设置命令行参数`--model <model/id/or/path>`。
- LoRA训练后未合并的模型，`--modelmodel/id/or/path>`指定基模路径的同时设置`--adapters <path/to/adapter>`。

### Q2: SWIFT如何使用数据集进行推理？推理结果保存在哪儿？
- 通过`--val_dataset <path/to/val_dataset>`指定数据集。如果想对训练中切分的验证集进行推理可以设置参数`--load_data_args true`。
- 推理结果保存路径通过`--result_path <your/path>`设置，日志中会打印路径。详见文档[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)。
- 如果需要保留推理数据集中非messages字段的额外字段，请设置`--remove_unused_columns false`。

### Q3: SWIFT如何设置批量推理？
如果infer_backend为transformers，通过设置命令行参数`--max_batch_size 16`，但需要注意该参数设置的是每张卡上的batch_size，不是全局的。或参考[demo](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py)。

### Q4: SWIFT如何设置流式推理？
使用`--stream true`，此时推理结果将逐条写入jsonl文件。<br>
注意：
- 流式推理不支持ddp。

### Q5: vLLM和SGLang推理后端相关的问题
- 对于LoRA训练的模型是否需要合并，请查看vLLM和SGLang文档，如果支持LoRA推理则不需要在推理前合并。
- SGLang推理目前不支持多模态。

### Q6: 生成参数相关的问题
temperature等参数默认从generation_config.json中读取。也可以通过`--temperature 0`或者`--top_k 1`显式设置来取消推理随机性。

### Q7: 如何将system_prompt置空？命令行不设置system参数，但是它会加上默认的system。
显性设置`--system ''`。

### Q8: 推理时如何计算acc/rouge等指标？
使用`--metric`，具体详情在[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id14)中搜索该参数。

### Q9: 模型推理的时候如果需要在特定前缀下继续推理的话是设置哪个参数？
使用参数`--response_prefix`。

### Q10: 数据answer里面已经包含了部分prompt，希望补全answer，应该怎么修改inference？
```text
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "answer1, "}]}
```
参考[examples/infer/demo_agent](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py)。

### Q11: 多模态模型推理时如何限制最大像素，以减少显存占用？
设置命令行参数`--max_pixels xxx`、环境变量`MAX_PIXELS=xxx`、或特定模型参数`--model_kwargs '{"max_pixels": xxx}'`。其中环境变量仅对文档中对应的模型生效，详见文档[特定模型参数](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id19)。

### Q12: SWIFT推理如何输出概率值logprobs参数？
命令行推理设置`--logprobs true`，python脚本推理设置
```shell
request_config = RequestConfig(..., logprobs=True, top_logprobs=2)
```
具体可以参考[test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/infer/test_logprobs.py)。

### Q13: SWIFT推理如何输出last_hidden_state？
没有参数可以直接使用，可以在[这里](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/grpo_trainer.py)参考GRPO trainer的`_get_last_hidden_state`方法。

### Q14: transformers，vllm，ollama等推理结果不一致问题
SWIFT的template是对齐transformers的。检查推理参数是否对其。此外，VllmEngine和TransformersEngine是有差异的。

### Q15: embedding/reranker模型推理
- embedding模型推理参考这里的[例子](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_embedding.py)。
- reranker模型推理参考这里的[例子](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_reranker.py)。

### Q16: 使用python脚本推理时，如何使用cpu?
设置环境变量，`os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`。

### Q17: 使用swift infer命令进行推理，支持多机推理吗？
如果单节点放得下模型，外面封装k8s就行。如果单节点放不下就不支持。

### Q18: swift sample的时候，是否支持batch？
这个[脚本](https://github.com/modelscope/ms-swift/blob/main/examples/train/rft/rft.py)，可以用多进程对数据集拆分采样。

### Q19: 特殊模型依赖版本相关问题
- Qwen2-Audio推理结果出现混乱，请使用transformers4.48。
- transformers4.55.2训练的LoRA不能使用小于4.52的版本加载，详见[issue#5440](https://github.com/modelscope/ms-swift/issues/5440)。
- swift对不同版本的qwen-vl-utils做了兼容，使用qwen2.5-vl和qwen3-vl模型时不需要切换该依赖版本。

### Q20: safetensors_rust.SafetensorError: Error while deserializing header:MetadataIncompleteBuffer
模型权重损坏。

### Q21: vLLM 报错如下：
```shell
ValueError: the decoder prompt contains a(n) video item with length 16758, which exceeds the pre-allocated encoder cache size 16384. please reduce the input size or increase the encoder cache size by setting --limit-mm-per-prompt at startup.
```
这通常是多模态输入过长，超过了vLLM预分配的encoder cache size导致的。可以通过 `--limit_mm_per_prompt` 调整encoder cache size；另一个可行的解决方法是通过在Swift cli中传入：
```shell
--vllm_engine_kwargs '{"max_num_batched_tokens": 20000}'
```
来增大 `max_num_batched_tokens`间接影响encoder cache size的分配。

## 导出

### Q1: autoawq相关的报错
- 如果推理没有涉及AWQ量化模型，但出现了autoawq相关的报错，可以尝试卸载autoawq再进行推理。
- 不支持AWQ量化的模型，可以尝试用GPTQ进行量化。

### Q2: SWIFT量化模型时，一张卡上放不下模型的情况
尝试设置`--device_map cpu`；或者多卡加载模型，单卡量化。

### Q3: 用swift export对qwen2.5 72B模型进行gptq int4量化，max model length用的是默认值32768，给的校准数据集有128个样本，但是量化的时候报错了，报错日志如下：
```shell
factorization could not be completed because the input is not positive-definite(the leading minor of order 18145 is not pisitive-definite)
```
海森矩阵不正定的问题，试试其他的数据集。

### Q4: swift export的时候传入自定义的template_type,是否可以永久改掉template_type？
不会被修改,swift中的template是定义在swift内部的,不是以jinja方式保存的。

### Q5: 模型训练完能直接转gguf格式吗？
目前只支持导出ModelFile，详见[命令行参数文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html#id17)关于导出参数部分。

## 部署

### Q1: SWIFT部署如何设置模型？
- 全参数训练的模型、LoRA训练后合并的模型或者从model hub下载的模型，设置命令行参数`--model <model/id/or/path>`。
- LoRA训练后未合并的模型，`--modelmodel/id/or/path>`指定基模路径的同时设置`--adapters <path/to/adapter>`。

### Q2: SWIFT如何进行多卡部署？
详见[例子](https://github.com/modelscope/ms-swift/tree/main/examples/deploy)。如果是transformers engine，不支持DDP，不能多卡部署。此外，不支持异构部署，如不同型号的显卡、各显卡设置不同的存储占比等。

### Q3: 通过--system参数指定system prompt与数据集中每个数据前加system prompt以及template的system prompt是否选一操作即可？这些方式对模型来说，优先级是否一样？
system优先级：数据集中的>命令行的>template中默认的。

### Q4: 客户端多模态输入相关问题
- 客户端传入图片、音频等，见[客户端例子](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm)。
- 如果图片url非法，可以通过设置环境变量`SWIFT_TIMEOUT`，或者`InferClient`中传参数来设置请求的超时时间。

### Q5: 生成参数设置相关问题
- 推理生成参数（temperature 等）部署时可设默认值，客户端每次请求可动态覆盖；
- 引擎/部署参数（TP 数、显存占比、最大长度）只能在部署启动时设置，运行后不可改。

### Q6: SWIFT部署的模型怎么设置流式生成？
是由客户端控制的，详情请查看[examples/deploy/client](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client)。

### Q7: SWIFT部署如何输出token的概率？
首先服务端需要设置`--logprobs true`，其次客户端需要传以下参数：
```shell
request_config = RequestConfig(..., logprobs=True, top_logprobs=2)
```

### Q8: thinking相关问题
如果需要禁止思考，目前只能在swift deploy启动的时候禁止thinking。可以查看这个[issue](https://github.com/modelscope/ms-swift/issues/4030)。

### Q9: 如何实现一次输出多个结果？
在`RequestConfig`中传入参数`n`，如下所示：
```shell
response = client.infer([request], request_config=RequestConfig(
    n=3,              # 生成 3 条
    temperature=0.8,  # 需要有随机性才能产生不同结果
))
# response 包含 3 条不同的回答
```

### Q10: 指定--infer_backend vllm，和直接使用vllm部署推理结果有区别
- 推理结果相差较多，可能是template没对齐。
- 推理速度相差较多，可能是图像分辨率不一致。
- SWIFT默认使用V1 engine，可以通过环境变量`VLLM_USE_V1=1`控制切换。

### Q11: 特殊模型和依赖版本相关问题
- 如果遇到报错没有`model.language_model.embed_tokens.weight`，可能由于训练-推理的transformers版本不一致导致。
- qwen2.5使用fp16推理如果遇到返回乱码，尝试bf16。

### Q12: Qwen2-7B base 模型部署后为什么不能用 chat.completions 而要用 completions？
base 模型没有经过对话格式训练，它不认识 <|im_start|>user<|im_end|> 这类 chat special token。SWIFT框架做了处理，base模型也可以用client.chat.completions.create，不过这个是兼容行为，本质上还是把 messages 拼成纯文本做续写。

## 评测

ms-swift的eval能力使用了魔搭社区评测框架EvalScope, 复杂能力请直接使用[EvalScope框架](https://evalscope.readthedocs.io/zh-cn/latest/get_started/introduction.html)。

### Q1: SWIFT支持的评测集有哪些？以及如何使用自定义评测集？
标准评测集和用户自定义评测集的使用详见[评测文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/Evaluation.html)。

### Q2: 官方支持的评测数据集手动下载后，swift eval能配置本地路径评测吗？
离线评测请参考EvalScope文档[快速上手](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html)。

### Q3: eval微调后的模型，总是会在固定的百分比停掉，但是vllm服务一直在正常运行
客户端请求超过了默认超时时间，连接被断开了。可以将`SWIFT_TIMEOUT`环境变量设置为-1，关闭超时断开。

### Q4: 评估的时候可不可以控制数据集条数？
配置参数`--eval_limit`，该参数控制了每个subset的条数，比如mmlu有50多个subset，每个subset limit10条，总共500多条。

### Q5: 模型最多生成1024token就结束了，这个如何修改？尝试设置`--max_new_tokens`5000不起作用
`--max_new_tokens`是推理参数，不是评测参数。评测时的生成长度由`--eval_generation_config`控制，需要在这个参数里设置`max_new_tokens`。
```shell
--eval_generation_config '{"max_new_tokens": 5000}'
```

### Q6: `--eval_backend OpenCompass`不支持自定义数据集吗？报错如下：
```shell
ValueError: eval_dataset: /mnt/workspace/data.jsonl is not supported.
eval_backend: OpenCompass supported datasets: ['C3', 'summedits', 'WiC', 'csl', 'lambada', 'mbpp', 'hellaswag', 'ARC_e', 'math', 'nq', 'race', 'MultiRC', 'cmb', 'ceval', 'GaokaoBench', 'mmlu', 'winogrande', 'tnews', 'triviaqa', 'CB', 'cluewsc', 'humaneval', 'AX_g', 'DRCD', 'RTE', 'ocnli_fc', 'gsm8k', 'obqa', 'ReCoRD', 'Xsum', 'ocnli', 'WSC', 'siqa', 'agieval', 'piqa', 'cmnli', 'cmmlu', 'eprstmt', 'storycloze', 'AX_b', 'afqmc', 'strategyqa', 'bustm', 'BoolQ', 'COPA', 'ARC_c', 'PMMEval', 'chid', 'CMRC', 'lcsts']
```
OpenCompass只支持其预定义的标准评测集，不支持自定义数据集，用native可以自定义模式。

### Q7: evalscope原生是可以生成报告的，其他后端如opencompass是否同样支持？
目前只支持native的可视化，其他后端还不支持。

### Q8: 评测ifeval报错：
```shell
[Errno 20] Not a directory: '/root/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'
```
需要解压`unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`。

### Q9: eval_backend='OpenCompass'，怎么指定离线数据集路径？
查看[数据准备教程](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/opencompass_backend.html#id3)，下载数据集并解压。不用指定`dataset-args`，将数据集文件夹（即data文件夹）放置在当前工作路径下即可，OpenCompass会自动识别。

### Q10: 报错：
```shell
unzip: cannot find or open /root/nltk_data/tokenizers/punkt_tab.zip, /root/nltk_data/tokenizers/punkt_tab.zip.zip or /root/nltk_data/tokenizers/punkt_tab.zip.ZIP
```
这是下载nltk的依赖失败，手动下载[punkt_tab.zip](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip)，解压到`~/nltk_data/tokenizers`下面。

### Q11: 是否可以指定llm作为judge, 参数应该怎么传进去？
支持的，参数传递如下所示：
```shell
--extra_eval_args '{"judge-model-args": {"api_key": "xxx", "api_url": "http://xxx/v1", "model_id": "qwen-72b"}}'
```

### Q12: 在执行eval的时候出现了多卡显存分配不均，报错如下：
```shell
NPROC_PER_NODE=8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\ MAX_PIXELS=802816\ swift eval\
--model "$MODEL_PATH” \$EXTRA_ARGS \
--eval_backend Native \ --infer_backend transformers\ --device_map auto \
--eval_limit"$EVAL_LIMIT"\ --eval_dataset general_qa\
--dataset_args "{\"general_qa\": {\"local_path\": \"${DATA_PATH}\", \"subset_list\": [\"${SUBSET_NAME}\"]}}" \ --host 127.0.0.1\> "$LOG_FILE" 2>&1
```
swift eval不支持DDP方式启动。

### Q13: 哪里可以看到swift评测的时候送入的query除了问题之外还有哪些额外的字段呢？
最简单的方法是看输出的reviews文件中的input字段，是输入给模型的内容转换后的Markdown格式。<br>
如果backend是opencompass的话没有这些输出，就需要用native backend。
