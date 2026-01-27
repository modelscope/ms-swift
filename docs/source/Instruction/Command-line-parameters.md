# 命令行参数

命令行参数的介绍会分为基本参数，原子参数、集成参数和特定模型参数。**命令行最终使用的参数列表为集成参数。集成参数继承自基本参数和一些原子参数**。特定模型参数是针对于具体模型的参数，可以通过`--model_kwargs'`或者环境变量进行设置。Megatron-SWIFT命令行参数介绍可以在[Megatron-SWIFT训练文档](../Megatron-SWIFT/Command-line-parameters.md)中找到。

**提示：**
- 命令行传入list使用空格隔开即可。例如：`--dataset <dataset_path1> <dataset_path2>`。
- 命令行传入dict使用json。例如：`--model_kwargs '{"fps_max_frames": 12}'`。
- 带🔥的参数为重要参数，刚熟悉ms-swift的用户可以先关注这些命令行参数。

## 基本参数

- 🔥tuner_backend: 可选为'peft'，'unsloth'。默认为'peft'。
- 🔥tuner_type: 可选为'lora'、'full'、'longlora'、'adalora'、'llamapro'、'adapter'、'vera'、'boft'、'fourierft'、'reft'。默认为'lora'。（**在ms-swift3.x中参数名为`train_type`**）
- 🔥adapters: 用于指定adapter的id/path的list，默认为`[]`。该参数通常用于推理/部署命令，例如：`swift infer --model '<model_id_or_path>' --adapters '<adapter_id_or_path>'`。该参数偶尔也用于断点续训，该参数与`resume_from_checkpoint`的区别在于，**该参数只读取adapter权重**，而不加载优化器和随机种子，并不跳过已训练的数据集部分。
  - `--model`与`--adapters`的区别：`--model`后接完整权重的目录路径，内包含model/tokenizer/config等完整权重信息，例如`model.safetensors`。`--adapters`后接增量adapter权重目录路径的列表，内涵adapter的增量权重信息，例如`adapter_model.safetensors`。
- 🔥external_plugins: 外部`plugin.py`文件列表，这些文件会被额外加载（即对模块进行`import`）。默认为`[]`。你可以传入自定义模型、对话模板和数据集注册的`.py`文件路径，参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/custom/sft.sh)；或者自定义GRPO的组件，参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_reward_func.sh)。
- seed: 全局随机种子，默认为42。
  - 注意：该随机种子与控制数据集随机的`data_seed`互不影响。
- model_kwargs: 特定模型可传入的额外参数，该参数列表会在训练/推理时打印日志进行提示。例如`--model_kwargs '{"fps_max_frames": 12}'`。你也可以通过环境变量的方式设置，例如`FPS_MAX_FRAMES=12`。默认为None。
  - 注意：**若你在训练时指定了特定模型参数，请在推理时也设置对应的参数**，这可以提高训练效果。
  - 特定模型参数的含义可以在对应模型官方repo或者其推理代码中找到相应含义。ms-swift引入这些参数以确保训练的模型与官方推理代码效果对齐。
- load_args: 当指定`--resume_from_checkpoint`、`--model`、`--adapters`会读取保存文件中的`args.json`，读取的keys查看[base_args.py](https://github.com/modelscope/ms-swift/blob/main/swift/arguments/base_args/base_args.py)。推理和导出时默认为True，训练时默认为False。该参数通常不需要修改。
- load_data_args: 如果将该参数设置为True，则会额外读取`args.json`中的数据参数。默认为False。**该参数通常用于推理时对训练中切分的验证集进行推理**，例如：`swift infer --adapters xxx --load_data_args true --stream true --max_new_tokens 512`。
- use_hf: 控制模型下载、数据集下载、模型推送使用ModelScope还是HuggingFace。默认为False，使用ModelScope。
- hub_token: hub token. modelscope的hub token可以查看[这里](https://modelscope.cn/my/myaccesstoken)。默认为None。
- ddp_timeout: 默认为18000000，单位为秒。
- ddp_backend: 可选为"nccl"、"gloo"、"mpi"、"ccl"、"hccl"、"cncl"、"mccl"。默认为None，进行自动选择。
- ignore_args_error: 用于兼容jupyter notebook。默认为False。

### 模型参数
- 🔥model: [模型id](https://modelscope.cn/models)或模型本地路径。默认为None。
- 🔥model_type: 模型类型。**我们将相同的模型架构、模型加载过程、template定义为一个`model_type`**。默认为None，即**根据`--model`的后缀和config.json中的'architectures'属性进行自动选择**。对应模型的model_type可以在[支持的模型列表](./Supported-models-and-datasets.md)中找到。
  - 注意：ms-swift中model_type的概念与`config.json`中的model_type不同。
  - 自定义模型通常需要自行注册`model_type`和`template`，具体可以参考[自定义模型文档](../Customization/Custom-model.md)。
- model_revision: 模型版本，默认为None。
- task_type: 默认为'causal_lm'。可选为'causal_lm'、'seq_cls'、'embedding'、'reranker'和'generative_reranker'。seq_cls的例子可以查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls)，embedding的例子查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding)。
  - 若设置为'seq_cls'，你通常需要额外设置`--num_labels`和`--problem_type`。
- 🔥torch_dtype: 模型权重的数据类型，支持`float16`,`bfloat16`,`float32`。默认为None，从'config.json'文件中读取。
- attn_impl: attention类型，可选项为'sdpa', 'eager', 'flash_attn', 'flash_attention_2', 'flash_attention_3'等。默认使用None，读取'config.json'。
  - 注意：这几种attention实现并不一定都支持，这取决于对应模型transformers实现的支持情况。
  - 若设置为'flash_attn'（兼容旧版本），则使用'flash_attention_2'。
- experts_impl: 专家实现类型，可选项为'grouped_mm', 'batched_mm', 'eager'。默认为None。该特性需要"transformers>=5.0.0"。
- new_special_tokens: 需要新增的特殊tokens。默认为`[]`。例子参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens)。
  - 注意：你也可以传入以`.txt`结尾的文件路径，每行为一个special token。
- num_labels: 分类模型（即`--task_type seq_cls`）需要指定该参数。代表标签数量，默认为None。
- problem_type: 分类模型（即`--task_type seq_cls`）需要指定该参数。可选为'regression', 'single_label_classification', 'multi_label_classification'。默认为None，若模型为 reward_model 或 num_labels 为1，该参数为'regression'，其他情况，该参数为'single_label_classification'。
- rope_scaling: rope类型，你可以传入字符串，例如：`linear`、`dynamic`、`yarn`并结合传入`max_model_len`，ms-swift会自动设置对应的rope_scaling并覆盖'config.json'中的rope_scaling。或者你需要传入一个json字符串，例如`'{"factor":2.0, "type":"yarn"}'`，该值会直接覆盖'config.json'中的rope_scaling。默认为None。
- max_model_len: 如果使用`rope_scaling`并传入字符串，可以设置`max_model_len`，该参数用来计算rope的`factor`倍数。该参数默认为None。若为非None，该参数会**覆盖**'config.json'中的`max_position_embeddings`值。
- device_map: 模型使用的device_map配置，例如：'auto'、'cpu'、json字符串、json文件路径。该参数会**透传**入transformers的`from_pretrained`接口。默认为None，根据设备和分布式训练情况自动设置。
- max_memory: device_map设置为'auto'或者'sequential'时，会根据max_memory进行模型权重的device分配，例如：`--max_memory '{0: "20GB", 1: "20GB"}'`。默认为None。该参数会透传入transformers的`from_pretrained`接口。
- local_repo_path: 部分模型在加载时依赖于github repo，例如[deepseek-vl2](https://github.com/deepseek-ai/DeepSeek-VL2)。为了避免`git clone`时遇到网络问题，可以直接使用本地repo。该参数需要传入本地repo的路径, 默认为`None`。
- init_strategy: 加载模型时，初始化模型中所有未初始化的参数（自定义模型架构时）。可选为'zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal'。默认为None。


### 数据参数
- 🔥dataset: 数据集id或路径的list。默认为`[]`。每个数据集的传入格式为：`'数据集id or 数据集路径:子数据集#采样数量'`，其中子数据集和取样数据可选。本地数据集支持jsonl、csv、json、文件夹等。**hub端的开源数据集可以通过`git clone`到本地并将文件夹传入而离线使用**。自定义数据集格式可以参考[自定义数据集文档](../Customization/Custom-dataset.md)。你可以传入`--dataset <dataset1> <dataset2>`来使用多个数据集。
  - 子数据集: 该参数只有当dataset为ID或者文件夹时生效。若注册时指定了subsets，且只有一个子数据集，则默认选择注册时指定的子数据集，否则默认为'default'。你可以使用`/`来选择多个子数据集，例如：`<dataset_id>:subset1/subset2`。你也可以使用'all'来选择注册时指定的所有子数据集，例如：`<dataset_id>:all`。注册例子可以参考[这里](https://modelscope.cn/datasets/swift/garbage_competition)。
  - 采样数量: 默认使用完整的数据集。你可以通过设置`#采样数`对选择的数据集进行采样，例如`<dataset_path#1000>`。若采样数少于数据样本总数，则进行随机选择（不重复采样）。若采样数高于数据样本总数，则只额外随机采样`采样数%数据样本总数`的样本，数据样本重复采样`采样数//数据样本总数`次。注意：流式数据集（`--streaming true`）只进行顺序采样。若设置`--dataset_shuffle false`，则非流式数据集也进行顺序采样。
- 🔥val_dataset: 验证集id或路径的list。默认为`[]`。
- 🔥cached_dataset: 使用缓存数据集（使用`swift export --to_cached_dataset true ...`命令产生），避免大型数据集训练/推理时，tokenize过程占用gpu时间。该参数用于设置缓存训练数据集文件夹路径，默认为`[]`。例子参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/cached_dataset)。
  - 提示：在"ms-swift>=3.11"，cached_dataset只会在数据集中额外存储length字段（为避免存储压力），并过滤掉会报错的数据样本。在训练/推理时，支持`--max_length`参数进行超长数据过滤/裁剪以及`--packing`参数。数据实际预处理过程将在训练时同步进行，该过程和训练是重叠的，并不会影响训练速度。
  - cached_dataset在`ms-swift`和`Megatron-SWIFT`之间是通用的，且支持pt/sft/infer/rlhf（需"ms-swift>=3.11"），使用`--template_mode`设置训练类型；在"ms-swift>=3.12"，支持embedding/reranker/seq_cls任务，使用`--task_type`设置任务类型。
  - 在"ms-swift>=3.12"，支持了对cache_dataset进行采样的功能，语法为`<cached_dataset_path>#采样数`，支持采样数高于和少于样本数的情况，功能与实现参考`--dataset`的介绍。
- cached_val_dataset: 缓存验证数据集的文件夹路径，默认为`[]`。
- 🔥split_dataset_ratio: 不指定val_dataset时从训练集拆分验证集的比例，默认为0.，即不从训练集切分验证集。
  - 注意：该参数在"ms-swift<3.6"的默认值为0.01。
- data_seed: 数据集随机种子，默认为42。
- 🔥dataset_num_proc: 数据集预处理的进程数，默认为1。
  - 提示：纯文本模型建议将该值开大加速预处理速度。而多模态模型不建议开太大，这可能导致更慢的预处理速度（多模态模型若出现cpu利用率100%，但是处理速度极慢的情况，建议额外设置`OMP_NUM_THREADS`环境变量）。
- 🔥load_from_cache_file: 是否从缓存中加载数据集，默认为False。**建议在实际运行时设置为True，debug阶段设置为False**。你可以修改`MODELSCOPE_CACHE`环境变量控制缓存的路径。
  - 注意：该参数在"ms-swift<3.9"默认为True。
- dataset_shuffle: 是否对dataset进行随机操作。默认为True。
  - 注意：**CPT/SFT的随机包括两个部分**：数据集的随机，由`dataset_shuffle`控制；train_dataloader中的随机，由`train_dataloader_shuffle`控制。
- val_dataset_shuffle: 是否对val_dataset进行随机操作。默认为False。
- streaming: 流式读取并处理数据集，默认False。（流式数据集的随机并不彻底，可能导致loss波动剧烈。）
  - 注意：需要额外设置`--max_steps`，因为流式数据集无法获得其长度。你可以通过设置`--save_strategy epoch`并设置较大的max_steps来实现与`--num_train_epochs`等效的训练。或者，你也可以设置`max_epochs`确保训练到对应epochs时退出训练，并对权重进行验证和保存。
  - 注意：流式数据集可以跳过预处理等待，将预处理时间与训练时间重叠。流式数据集的预处理只在rank0上进行，并通过数据分发的方式同步到其他进程，**其通常效率不如非流式数据集采用的数据分片读取方式**。当训练的world_size较大时，预处理和数据分发将成为训练瓶颈。
- interleave_prob: 默认值为 None。在组合多个数据集时，默认使用datasets库的 `concatenate_datasets` 函数；如果设置了该参数，则会使用 `interleave_datasets` 函数。该参数通常用于流式数据集的组合，并会作为参数传入 `interleave_datasets` 函数中。该参数不对`--val_dataset`生效。
- stopping_strategy: 可选为"first_exhausted", "all_exhausted"，默认为"first_exhausted"。传入`interleave_datasets`函数中。该参数不对`--val_dataset`生效。
- shuffle_buffer_size: 该参数用于指定**流式数据集**的随机buffer大小，默认为1000。该参数只在`dataset_shuffle`设置为true时有效。
- download_mode: 数据集下载模式，包含`reuse_dataset_if_exists`和`force_redownload`，默认为'reuse_dataset_if_exists'。
  - 通常在使用hub端数据集报错时设置为`--download_mode force_redownload`。
- columns: 用于对数据集进行列映射，使数据集满足AutoPreprocessor可以处理的样式，AutoPreprocessor可以处理的数据集格式查看[自定义数据集文档](../Customization/Custom-dataset.md)。你可以传入json字符串，例如：`'{"text1": "query", "text2": "response"}'`，代表将数据集中的"text1"映射为"query"，"text2"映射为"response"，而query-response格式可以被AutoPreprocessor处理。默认为None。
- strict: 如果为True，则数据集只要某行有问题直接抛错，否则会丢弃出错数据样本。默认False。该参数通常用于排查错误。
- 🔥remove_unused_columns: 是否删除数据集中不被使用的列，默认为True。
  - 若该参数设置为False，则将额外的数据集列传递至trainer的`compute_loss`函数内，**方便自定义损失函数使用额外的数据集列**。
  - GPRO该参数的默认值为False。
- 🔥model_name: **仅用于自我认知任务**，只对`swift/self-cognition`数据集生效，替换掉数据集中的`{{NAME}}`通配符。传入模型中文名和英文名，以空格分隔，例如：`--model_name 小黄 'Xiao Huang'`。默认为None。
- 🔥model_author: 仅用于自我认知任务，只对`swift/self-cognition`数据集生效，替换掉数据集中的`{{AUTHOR}}`通配符。传入模型作者的中文名和英文名，以空格分隔，例如：`--model_author '魔搭' 'ModelScope'`。默认为None。
- custom_dataset_info: 自定义数据集注册的json文件路径，参考[自定义数据集](../Customization/Custom-dataset.md)和[内置'dataset_info.json'文件](https://github.com/modelscope/ms-swift/blob/main/swift/dataset/data/dataset_info.json)。默认为`[]`。

### 模板参数
- 🔥template: 对话模板类型。默认为None，自动选择对应model的template类型，对应关系参考[支持的模型列表](./Supported-models-and-datasets.md)。
- 🔥system: 自定义system字段，可以传入字符串或者**txt文件路径**。默认为None，使用注册template时的默认system。
  - 注意：数据集中的system**优先级**最高，然后是`--system`，最后是注册template时设置的`default_system`。
- 🔥max_length: 限制单数据集样本经过`tokenizer.encode`后的tokens最大长度，超过的数据样本会根据`truncation_strategy`参数进行处理（避免训练OOM）。默认为None，即设置为模型支持的tokens最大长度(max_model_len)。
  - 当PPO、GRPO、GKD和推理情况下，`max_length`代表`max_prompt_length`。
- truncation_strategy: 如果单样本的tokens超过`max_length`如何处理，支持'delete'、'left'、'right'和'split'，代表删除、左侧裁剪、右侧裁剪和切成多条数据样本，默认为'delete'。
  - 注意：`--truncation_strategy split`只支持预训练时使用，即`swift/megatron pt`场景下，需"ms-swift>=3.11"，该策略会将超长字段切成多条数据样本，从而避免tokens浪费。（该特性不兼容cached_dataset）
  - 注意：若多模态模型的训练时将'truncation_strategy'设置为`left`或`right`，**ms-swift会保留所有的image_token等多模态tokens**，这可能会导致训练时OOM。
- 🔥max_pixels: 多模态模型输入图片的最大像素数（H\*W），将超过该限制的图像进行缩放（避免训练OOM）。默认为None，不限制最大像素数。
  - 注意：该参数适用于所有的多模态模型。而Qwen2.5-VL特有的模型参数`MAX_PIXELS`（你可以在文档最下面找到）只针对Qwen2.5-VL模型。
- 🔥agent_template: Agent模板，确定如何将工具列表'tools'转换成'system'、如何在推理/部署时从模型回复中提取toolcall部分，以及确定'messages'中`{"role": "tool_call", "content": "xxx"}`, `{"role": "tool_response", "content": "xxx"}`的模板格式。可选为"react_en", "hermes", "glm4", "qwen_en", "toolbench"等，更多请查看[这里](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/mapping.py)。默认为None，根据模型类型进行自动选择。可以参考[Agent文档](./Agent-support.md)。
- norm_bbox: 控制如何缩放边界框（即数据集中的"bbox"，里面的数据为绝对坐标，参考[自定义数据集文档](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html#grounding)）。选项为'norm1000'和'none'。'norm1000'表示将bbox坐标缩放至千分位坐标，而'none'表示不进行缩放。默认值为None，将根据模型自动选择。
  - 当**图片在训练中发生缩放时**（例如设置了max_pixels参数），该参数也能很好进行解决。
- use_chat_template: 使用chat模板还是generation模板（generation模板通常用于预训练时）。默认为`True`。
  - 注意：`swift pt`默认为False，使用generation模板。该参数可以很好的**兼容多模态模型**。
- padding_side: 当训练`batch_size>=2`时的padding_side，可选值为'left'、'right'，默认为'right'。（推理时的batch_size>=2时，只进行左padding）。
  - 注意：PPO和GKD默认设置为'left'。
- 🔥padding_free: 将一个batch中的数据进行展平而避免数据padding，从而降低显存占用并加快训练（**同一batch的不同序列之间依旧是不可见的**）。默认为False。当前支持CPT/SFT/DPO/GRPO/KTO/GKD。
  - 注意：使用padding_free请结合`--attn_impl flash_attn`使用且"transformers>=4.44"，具体查看[该PR](https://github.com/huggingface/transformers/pull/31629)。（同packing）
  - **相较于packing，padding_free不需要额外的预处理时间，但packing的训练速度更快且显存占用更稳定**。
- 🔥loss_scale: 训练tokens的loss权重设置。默认为`'default'`。loss_scale包含3种基本策略：'default'、'last_round'、'all'，以及其他策略：'ignore_empty_think'以及agent需要的：'react'、'hermes'、'qwen'、'agentflan'、'alpha_umi'等，可选值参考[loss_scale模块](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/mapping.py)。ms-swift>=3.12 支持了基本策略和其他策略的混用，例如：`'default+ignore_empty_think'`，`'last_round+ignore_empty_think'`。若没有指定基本策略，则默认为'default'，例如：'hermes'与'default+hermes'等价。
  - 'default': 所有response（含history）以权重1计算交叉熵损失（**messages中的system/user/多模态tokens以及Agent训练中`tool_response`部分不计算损失**）。（**SFT默认为该值**）
  - 'last_round': 只计算最后一轮response的损失。在"ms-swift>=3.12"，最后一轮含义为最后一个"user"之后的所有内容，之前的含义只包含最后一个"assistant"。（**RLHF默认为该值**）
  - 'all': 计算所有tokens的损失。（**`swift pt`默认为该值**）
  - 'ignore_empty_think': 忽略空的`'<think>\n\n</think>\n\n'`损失计算。（满足正则匹配`'<think>\\s*</think>\\s*'`即可）。
  - 'react', 'hermes', 'qwen': 将`tool_call`部分的loss权重调整为2。
- sequence_parallel_size: 序列并行大小，默认是1。当前支持CPT/SFT/DPO/GRPO。训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel)。
- template_backend: 选择template后端，可选为'swift'、'jinja'，默认为'swift'。如果使用jinja，则使用transformers的`apply_chat_template`。
  - 注意：jinja的template后端只支持推理，不支持训练（无法确定损失计算的tokens范围）。
- response_prefix: response的前缀字符，该参数只在推理时生效。默认为None，根据enable_thinking参数和模版类型确定。
- enable_thinking: (ms-swift>=3.12) 该参数在推理时生效，代表是否开启thinking模式。默认为None，默认值由模板（模型）类型确定（思考/混合思考模板为True，非思考模板为False）。若enable_thinking为False，则增加非思考前缀，例如Qwen3-8B混合思考模型增加前缀`'<think>\n\n</think>\n\n'`，Qwen3-8B-Thinking则不增加前缀。若enable_thinking为True，则增加思考前缀，例如`'<think>\n'`。注意：该参数的优先级低于response_prefix参数。
  - 注意：对于思考模型（思考/混合思考）或显式开启enable_thinking，我们会在推理和训练时，对历史的思考内容进行删除（最后一轮的思考内容保留，即最后一个user信息后的内容）。若训练时的loss_scale基本策略不为last_round，例如为'default'，则不对历史的思考内容进行删除。
- add_non_thinking_prefix: (ms-swift>=3.12) 该参数只在训练时生效，代表是否对数据样本assistant部分不以思考标记`'<think>'`开头的数据样本增加非思考前缀（通常混合思考模型含非思考前缀）。该特性可以让swift内置的数据集可以训练混合思考模型。默认值为True。例如：例如Qwen3-8B混合思考模型的非思考前缀为`'<think>\n\n</think>\n\n'`，Qwen3-8B-Thinking/Instruct的非思考前缀为`''`。注意：训练时，loss_scale的基本策略为last_round，则只对最后一轮做此修改；否则，例如为'default'、'all'，则对每一轮数据做此修改。若设置为False，则不对数据样本增加非思考前缀。

### 生成参数
参考[generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)文档。

- 🔥max_new_tokens: 推理最大生成新tokens的数量。默认为None，无限制。
- temperature: 温度参数，温度越高，输出的随机性越大。默认为None，读取'generation_config.json'。
  - 你可以设置`--temperature 0`或者`--top_k 1`以取消推理随机性。
- top_k: top_k参数，保留概率最高的top_k数量 tokens用于生成，默认为None。读取'generation_config.json'。
- top_p: top_p参数，保留概率最高的累计概率达到 top_p 的tokens用于生成，默认为None。读取generation_config.json。
- repetition_penalty: 重复惩罚参数。1.0 表示不进行惩罚。默认为None，读取generation_config.json。
- num_beams: beam search的并行保留数量，默认为1。
- 🔥stream: 流式输出，默认为`None`，即使用交互式界面时为True，数据集批量推理时为False。
  - "ms-swift<3.6"stream默认值为False。
- stop_words: 除了eos_token外额外的停止词，默认为`[]`。
  - 注意：eos_token会在输出respsone中被删除，额外停止词会在输出中保留。
- logprobs: 是否输出logprobs，默认为False。
- top_logprobs: 输出top_logprobs的数量，默认为None。
- structured_outputs_regex: 结构化输出（引导解码）的正则表达式模式。设置后，模型生成将被约束为匹配指定的正则表达式模式。仅在`infer_backend`为`vllm`时生效。默认为`None`。

### 量化参数
以下为加载模型时量化的参数，具体含义可以查看[量化](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)文档。这里不包含`swift export`中涉及的`gptq`、`awq`量化参数。

- 🔥quant_method: 加载模型时采用的量化方法，可选项为'bnb'、'hqq'、'eetq'、'quanto'和'fp8'，默认为None。
  - 若对awq/gptq量化模型进行qlora训练，则不需要设置额外`quant_method`等量化参数。
- 🔥quant_bits: 量化bits数，默认为None。
- hqq_axis: hqq量化axis，默认为None。
- bnb_4bit_compute_dtype: bnb量化计算类型，可选为`float16`、`bfloat16`、`float32`。默认为None，设置为`torch_dtype`。
- bnb_4bit_quant_type: bnb量化类型，支持`fp4`和`nf4`，默认为`nf4`。
- bnb_4bit_use_double_quant: 是否使用双重量化，默认为`True`。
- bnb_4bit_quant_storage: bnb量化存储类型，默认为None。

### RAY参数

- use_ray: boolean类型。是否使用ray，默认为`False`
- ray_exp_name: ray实验名字，这个字段会用作cluster和worker名称前缀，可以不填
- device_groups: 字符串（jsonstring）类型。在使用ray时，该字段必须配置，具体可以查看[ray文档](Ray.md)。

### yaml支持

- config: 可以使用config代替命令行参数，例如：

```shell
swift sft --config demo.yaml
```

demo.yaml的内容为具体命令行配置：

```yaml
# Model args
model: Qwen/Qwen2.5-7B-Instruct
dataset: swift/self-cognition
...

# Train args
output_dir: xxx/xxx
gradient_checkpointing: true

...
```

## 原子参数

### Seq2SeqTrainer参数

该参数列表继承自transformers `Seq2SeqTrainingArguments`，ms-swift对其默认值进行了覆盖。未列出的请参考[HF官方文档](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments)。

- 🔥output_dir: 默认为None，设置为`'output/<model_name>'`。
- 🔥gradient_checkpointing: 是否使用gradient_checkpointing，默认为True。该参数可以显著降低显存占用，但降低训练速度。
- 🔥vit_gradient_checkpointing: 多模态模型训练时，是否对vit部分开启gradient_checkpointing。默认为None，即设置为`gradient_checkpointing`。例子参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/vit_gradient_checkpointing.sh)。
  - 注意：多模态模型且是LoRA训练时，当设置了`--freeze_vit false`，且命令行中出现以下警告：`UserWarning: None of the inputs have requires_grad=True. Gradients will be None`，请设置`--vit_gradient_checkpointing false`，或提相关issue。全参数训练则不会出现该问题。（如果RLHF LoRA训练中，ref_model抛出来的警告，则是正常的）
- 🔥deepspeed: 默认为None。可以设置为'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'来使用ms-swift内置的deepspeed配置文件。你也可以传入自定义deepspeed配置文件的路径。
- zero_hpz_partition_size: 默认为None，这个参数是ZeRO++的特性，即node内模型分片，node间数据分片，如果遇到grad_norm NaN，请尝试使用`--torch_dtype float16`。
- deepspeed_autotp_size: DeepSpeed张量并行大小，默认为1。使用DeepSpeed AutoTP时需将参数`--deepspeed`设置为'zero0'、'zero1'或'zero2'。（注意：该功能只支持全参数）
- 🔥fsdp: FSDP2分布式训练配置。默认为None。可以设置为'fsdp2'来使用ms-swift内置的FSDP2配置文件。你也可以传入自定义FSDP配置文件的路径。FSDP2是PyTorch原生的分布式训练方案，与DeepSpeed二选一使用。
- 🔥per_device_train_batch_size: 默认值1。
- 🔥per_device_eval_batch_size: 默认值1。
- 🔥gradient_accumulation_steps: 梯度累加。**默认为None，即设置gradient_accumulation_steps使得total_batch_size>=16**。total_batch_size等于`per_device_train_batch_size * gradient_accumulation_steps * world_size`。在GRPO训练中，默认为1。
  - 在CPT/SFT训练中，梯度累加的训练效果等价使用更大的batch_size，但在RLHF训练中，训练效果并不等价。
- weight_decay: weight衰减系数，默认值0.1。
- adam_beta1: 默认为0.9。
- adam_beta2: 默认为0.95。
- 🔥learning_rate: 学习率，**全参数训练默认为1e-5，LoRA训练等tuners为1e-4**。
  - 提示：若要设置`min_lr`，您可以传入参数`--lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'`。
- 🔥vit_lr: 当训练多模态大模型时，该参数指定vit的学习率，默认为None，等于learning_rate。通常与`--freeze_vit`、`--freeze_aligner`参数结合使用。
  - 提示：在日志中打印的"learning_rate"为`param_groups[0]`的学习率，其中param_groups的顺序依次是vit, aligner, llm（若含可训练参数）。
- 🔥aligner_lr: 当训练多模态大模型时，该参数指定aligner的学习率，默认为None，等于learning_rate。
- lr_scheduler_type: lr_scheduler类型，默认为'cosine'。
- lr_scheduler_kwargs: lr_scheduler其他参数。默认为None。
- gradient_checkpointing_kwargs: 传入`torch.utils.checkpoint`中的参数。例如设置为`--gradient_checkpointing_kwargs '{"use_reentrant": false}'`。默认为None。
  - 注意：当使用DDP而不使用deepspeed/fsdp，且gradient_checkpointing_kwargs为None，会默认设置其为`'{"use_reentrant": false}'`而避免出现报错。
- full_determinism: 确保训练中获得可重现的结果，注意：这会对性能产生负面影响。默认为False。
- 🔥report_to: 默认值为`tensorboard`。你也可以指定`--report_to tensorboard wandb swanlab`、`--report_to all`。
  - 如果你指定了`--report_to wandb`，你可以通过`WANDB_PROJECT`设置项目名称，`WANDB_API_KEY`指定账户对应的API KEY。
- logging_first_step: 是否记录第一个step的日志，默认为True。
- logging_steps: 日志打印间隔，默认为5。
- router_aux_loss_coef: 用于moe模型训练时，设置 aux_loss 的权重，默认为`0.`。
  - 注意：在"ms-swift==3.7.0"，其默认为None，从config.json中读取，该行为在"ms-swift>=3.7.1"被修改。
- enable_dft_loss: 是否在SFT训练中使用[DFT](https://arxiv.org/abs/2508.05629) (Dynamic Fine-Tuning) loss，默认为False。
- enable_channel_loss: 启用channel loss，默认为`False`。你需要在数据集中准备"channel"字段，ms-swift会根据该字段分组统计loss（若未准备"channel"字段，则归为默认`None` channel）。数据集格式参考[channel loss](../Customization/Custom-dataset.md#channel-loss)。channel loss兼容packing/padding_free/loss_scale等技术。
  - 注意：该参数为"ms-swift>=3.8"新增，若要在"ms-swift<3.8"使用channel loss，请查看v3.7文档。
- logging_dir: tensorboard日志保存路径。默认为None，即设置为`f'{self.output_dir}/runs'`。
- 🔥predict_with_generate: 验证时使用生成式的方式，默认为False。
- metric_for_best_model: 默认为None，即当`predict_with_generate`设置为False时，设置为'loss'，否则设置为'rouge-l'（在PPO训练时，不进行默认值设置；GRPO训练设置为'reward'）。
- greater_is_better: 默认为None，即当`metric_for_best_model`含'loss'时，设置为False，否则设置为True。
- max_epochs: 训练到`max_epochs`时强制退出训练，并对权重进行验证和保存。该参数在使用流式数据集时很有用。默认为None。

其他重要参数：
- 🔥num_train_epochs: 训练的epoch数，默认为3。
- 🔥save_strategy: 保存模型的策略，可选为'no'、'steps'、'epoch'，默认为'steps'。
- 🔥save_steps: 默认为500。
- 🔥eval_strategy: 评估策略。默认为None，跟随`save_strategy`的策略。
  - 若不使用`val_dataset`和`eval_dataset`且`split_dataset_ratio`为0，则默认为'no'。
- 🔥eval_steps: 默认为None，如果存在评估数据集，则跟随`save_steps`的策略。
- eval_on_start: 是否在训练前执行一次评估步骤，以确保验证步骤能正常工作。默认为False。
- 🔥save_total_limit: 最多保存的checkpoint数，会将过期的checkpoint进行删除。默认为None，保存所有的checkpoint。
- max_steps: 最大训练的steps数。在数据集为流式时需要被设置。默认为-1。
- 🔥warmup_ratio: 默认为0.。
- save_on_each_node: 在每一个节点都进行权重保存。默认为False。该参数在多机训练时需要被考虑。
  - 提示：在多机训练时，通常将`output_dir`设置为节点共享目录，因此无需额外设置该参数。
- save_only_model: 是否只保存模型权重而不包含优化器状态，随机种子状态等内容，这在全参数训练时可以减少保存的时间消耗和空间占用。默认为False。
- 🔥resume_from_checkpoint: 断点续训参数，指定checkpoint路径。默认为None。
  - 提示：**断点续训请保持其他参数不变，额外增加`--resume_from_checkpoint checkpoint_dir`**。权重等信息将在trainer中读取。
  - 注意: resume_from_checkpoint会读取模型权重，优化器状态，随机种子，并从上次训练的steps继续开始训练。你可以指定`--resume_only_model`只读取模型权重。
- resume_only_model: 默认为False。如果在指定resume_from_checkpoint的基础上，将该参数设置为True，则仅resume模型权重，而忽略优化器状态和随机种子。
  - 注意：在"ms-swift>=3.7"，**resume_only_model默认将进行数据跳过**，此行为可通过 `ignore_data_skip` 参数控制。若要恢复"ms-swift<3.7"的行为，请设置`--ignore_data_skip true`。
- ignore_data_skip: 当设置`resume_from_checkpoint`和`resume_only_model`时，该参数控制是否跳过已经训练的数据，并将epoch和迭代数等训练状态进行恢复。默认为False。若设置为True，则将不加载训练状态并不进行数据跳过，将从迭代数0开始训练。
- 🔥ddp_find_unused_parameters: 默认为None。
- 🔥dataloader_num_workers: 默认为None，若是windows平台，则设置为0，否则设置为1。
- dataloader_pin_memory: 默认为True。
- dataloader_persistent_workers: 默认为False。
- dataloader_prefetch_factor: 默认为None。若 `dataloader_num_workers > 0`，则设置为2。每个工作进程预先加载的批次数量。2 表示所有工作进程总共会预取 2 * num_workers 个批次。
  - 在"ms-swift<3.12"，默认值为10，该值可能导致内存不足。
- train_dataloader_shuffle: CPT/SFT训练的dataloader是否随机，默认为True。该参数对IterableDataset无效（即对流式数据集失效）。IterableDataset采用顺序的方式读取。
- optim: 优化器，默认值为 `"adamw_torch"` (对于 torch>=2.8 为 `"adamw_torch_fused"`)。完整的优化器列表请参见 [training_args.py](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) 中的 `OptimizerNames`。
- optim_args: 提供给优化器的可选参数，默认为None。
- group_by_length: (ms-swift>=3.12) 是否在训练数据集中将长度大致相同的样本分组在一起（有随机因素），以最小化填充并确保各节点与进程的负载均衡以提高效率。默认为False。具体算法参考`transformers.trainer_pt_utils.get_length_grouped_indices`。
- 🔥neftune_noise_alpha: neftune添加的噪声系数。默认为0，通常可以设置为5、10、15。
- 🔥use_liger_kernel: 是否启用[Liger](https://github.com/linkedin/Liger-Kernel)内核加速训练并减少显存消耗。默认为False。示例shell参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/liger)。
  - 注意：liger_kernel不支持device_map，请使用DDP/DeepSpeed进行多卡训练。liger_kernel目前只支持`task_type='causal_lm'`。
- average_tokens_across_devices: 是否在设备之间进行token数平均。如果设置为True，将使用all_reduce同步`num_tokens_in_batch`以进行精确的损失计算。默认为False。
- max_grad_norm: 梯度裁剪。默认为1.。
  - 注意：日志中的grad_norm记录的是裁剪前的值。
- push_to_hub: 推送checkpoint到hub。默认为False。
- hub_model_id: 默认为None。
- hub_private_repo: 默认为False。

### Tuner参数
- 🔥freeze_llm: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_llm设置为True会将LLM部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_llm设置为True将会取消在LLM部分添加LoRA模块。该参数默认为False。
- 🔥freeze_vit: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_vit设置为True会将vit部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_vit设置为True将会取消在vit部分添加LoRA模块。该参数默认为True。
  - 注意：**这里的vit不仅限于vision_tower, 也包括audio_tower**。若是Omni模型，若你只希望对vision_tower加LoRA，而不希望对audio_tower加LoRA，你可以修改[这里的代码](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554)。
- 🔥freeze_aligner: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_aligner设置为True会将aligner（也称为projector）部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_aligner设置为True将会取消在aligner部分添加LoRA模块。该参数默认为True。
- 🔥target_modules: 指定lora模块, 默认为`['all-linear']`。你也可以设置为module的后缀，例如：`--target_modules q_proj k_proj v_proj`。该参数不限于LoRA，可用于其他tuners。
  - 注意：在LLM和多模态LLM中，'all-linear'的行为有所不同。若是LLM则自动寻找除lm_head外的linear并附加tuner；**若是多模态LLM，则默认只在LLM上附加tuner，该行为可以被`freeze_llm`、`freeze_vit`、`freeze_aligner`控制**。
- 🔥target_regex: 指定lora模块的regex表达式，默认为`None`。如果该值传入，则target_modules参数失效。例如你可以设置`--target_regex '^(language_model).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'`，将符合该正则的模块指定为LoRA模块。该参数不限于LoRA，可用于其他tuners。
- target_parameters: 要替换为LoRA的参数名称列表。该参数的行为与 `target_modules` 类似，但传入的应是参数名称而不是模块名称。该特性需要安装"peft>=0.17.0"。例如，在 Hugging Face Transformers 中许多混合专家（MoE）层中，并未使用 `nn.Linear`，而是使用了 `nn.Parameter`。这时可以使用target_parameters参数实现。
- init_weights: 初始化weights的方法，LoRA可以指定为`true`、`false`、`gaussian`、`pissa`、`pissa_niter_[number of iters]`，Bone可以指定为`true`、`false`、`bat`。默认值`true`。
- 🔥modules_to_save: 在已附加tuner后，额外指定一部分原模型模块参与训练和存储。默认为`[]`。该参数不限于LoRA，可用于其他tuners。例如设置为`--modules_to_save embed_tokens lm_head`，在LoRA训练中解开embed_tokens和lm_head层进行训练，这两部分的权重信息最终会保存在`adapter_model.safetensors`中。

#### 全参
- freeze_parameters: 需要被冻结参数的前缀，默认为`[]`。
- freeze_parameters_regex: 需要被冻结参数的正则表达式，默认为None。
- freeze_parameters_ratio: 从下往上冻结的参数比例，默认为0。可设置为1将所有参数冻结，结合`trainable_parameters`设置可训练参数。
- trainable_parameters: 额外可训练参数的前缀，默认为`[]`。
- trainable_parameters_regex: 匹配额外可训练参数的正则表达式，默认为None。
  - 备注：`trainable_parameters`、`trainable_parameters_regex`的优先级高于`freeze_parameters`、`freeze_parameters_regex`和`freeze_parameters_ratio`。例如：当指定全参数训练时，会将所有模块设置为可训练的状态，随后根据`freeze_parameters`、`freeze_parameters_regex`、`freeze_parameters_ratio`将部分参数冻结，最后根据`trainable_parameters`、`trainable_parameters_regex`重新打开部分参数参与训练。

#### LoRA
- 🔥lora_rank: 默认为`8`。
- 🔥lora_alpha: 默认为`32`。
- lora_dropout: 默认为`0.05`。
- lora_bias: 默认为`'none'`，可以选择的值: 'none'、'all'。如果你要将bias全都设置为可训练，你可以设置为`'all'`。
- lora_dtype: 指定lora模块的dtype类型。支持'float16'、'bfloat16'、'float32'。默认为None，跟随peft行为。
- 🔥use_dora: 默认为`False`，是否使用`DoRA`。
- use_rslora: 默认为`False`，是否使用`RS-LoRA`。
- 🔥lorap_lr_ratio: LoRA+参数，默认值`None`，建议值为`10~16`。使用lora时额外指定该参数可使用lora+。

##### LoRA-GA
- lora_ga_batch_size: 默认值为 `2`。在 LoRA-GA 中估计梯度以进行初始化时使用的批处理大小。
- lora_ga_iters: 默认值为 `2`。在 LoRA-GA 中估计梯度以进行初始化时的迭代次数。
- lora_ga_max_length: 默认值为 `1024`。在 LoRA-GA 中估计梯度以进行初始化时的最大输入长度。
- lora_ga_direction: 默认值为 `ArB2r`。在 LoRA-GA 中使用估计梯度进行初始化时的初始方向。允许的值有：`ArBr`、`A2rBr`、`ArB2r` 和 `random`。
- lora_ga_scale: 默认值为 `stable`。LoRA-GA 的初始化缩放方式。允许的值有：`gd`、`unit`、`stable` 和 `weightS`。
- lora_ga_stable_gamma: 默认值为 `16`。当初始化时选择 `stable` 缩放时的 gamma 值。

#### FourierFt

FourierFt使用`target_modules`、`target_regex`、`modules_to_save`三个参数，含义见上面文档中的描述。额外参数包括：

- fourier_n_frequency: 傅里叶变换的频率数量, `int`类型, 类似于LoRA中的`r`. 默认值`2000`.
- fourier_scaling: W矩阵的缩放值, `float`类型, 类似LoRA中的`lora_alpha`. 默认值`300.0`.

#### BOFT

BOFT使用`target_modules`、`target_regex`、`modules_to_save`三个参数，含义见上面文档中的描述。额外参数包括：

- boft_block_size: BOFT块尺寸, 默认值4.
- boft_block_num: BOFT块数量, 不能和`boft_block_size`同时使用.
- boft_dropout: boft的dropout值, 默认0.0.

#### Vera

Vera使用`target_modules`、`target_regex`、`modules_to_save`三个参数，含义见上面文档中的描述。额外参数包括：

- vera_rank: Vera Attention的尺寸, 默认值256.
- vera_projection_prng_key: 是否存储Vera映射矩阵, 默认为True.
- vera_dropout: Vera的dropout值, 默认`0.0`.
- vera_d_initial: Vera的d矩阵的初始值, 默认`0.1`.

#### GaLore

- 🔥use_galore: 默认值False, 是否使用GaLore.
- galore_target_modules: 默认值None, 不传的情况下对attention和mlp应用GaLore.
- galore_rank: 默认值128, GaLore的rank值.
- galore_update_proj_gap: 默认值50, 分解矩阵的更新间隔.
- galore_scale: 默认值1.0, 矩阵权重系数.
- galore_proj_type: 默认值`std`, GaLore矩阵分解类型.
- galore_optim_per_parameter: 默认值False, 是否给每个Galore目标Parameter设定一个单独的optimizer.
- galore_with_embedding: 默认值False, 是否对embedding应用GaLore.
- galore_quantization: 是否使用q-galore. 默认值`False`.
- galore_proj_quant: 是否对SVD分解矩阵做量化, 默认`False`.
- galore_proj_bits: SVD量化bit数.
- galore_proj_group_size: SVD量化分组数.
- galore_cos_threshold: 投影矩阵更新的cos相似度阈值. 默认值0.4.
- galore_gamma_proj: 在投影矩阵逐渐相似后会拉长更新间隔, 本参数为每次拉长间隔的系数, 默认值2.
- galore_queue_size: 计算投影矩阵相似度的队列长度, 默认值5.

#### LISA

注意: LISA仅支持全参数，即`--tuner_type full`。

- 🔥lisa_activated_layers: 默认值`0`，代表不使用LISA，改为非0代表需要激活的layers个数，建议设置为2或8。
- lisa_step_interval: 默认值`20`，多少iter切换可反向传播的layers。

#### UNSLOTH

🔥unsloth无新增参数，对已有参数进行调节即可支持，例如：

```
--tuner_backend unsloth
--tuner_type full/lora
--quant_bits 4
```

#### LLAMAPRO

- 🔥llamapro_num_new_blocks: 默认值`4`, 插入的新layers总数.
- llamapro_num_groups: 默认值`None`, 分为多少组插入new_blocks, 如果为`None`则等于`llamapro_num_new_blocks`, 即每个新的layer单独插入原模型.

#### AdaLoRA

以下参数`tuner_type`设置为`adalora`时生效. adalora的`target_modules`等参数继承于lora的对应参数，但`lora_dtype`参数不生效。

- adalora_target_r: 默认值`8`, adalora的平均rank.
- adalora_init_r: 默认值`12`, adalora的初始rank.
- adalora_tinit: 默认值`0`, adalora的初始warmup.
- adalora_tfinal: 默认值`0`, adalora的final warmup.
- adalora_deltaT: 默认值`1`, adalora的step间隔.
- adalora_beta1: 默认值`0.85`, adalora的EMA参数.
- adalora_beta2: 默认值`0.85`, adalora的EMA参数.
- adalora_orth_reg_weight: 默认值`0.5`, adalora的正则化参数.

#### ReFT

以下参数`tuner_type`设置为`reft`时生效.

> 1. ReFT无法合并tuner
> 2. ReFT和gradient_checkpointing不兼容
> 3. 如果使用DeepSpeed遇到问题请暂时卸载DeepSpeed

- 🔥reft_layers: ReFT应用于哪些层上, 默认为`None`, 代表所有层, 可以输入层号的list, 例如reft_layers 1 2 3 4`
- 🔥reft_rank: ReFT矩阵的rank, 默认为`4`.
- reft_intervention_type: ReFT的类型, 支持'NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention', 'LobireftIntervention', 'DireftIntervention', 'NodireftIntervention', 默认为`LoreftIntervention`.
- reft_args: ReFT Intervention中的其他支持参数, 以json-string格式输入.

### vLLM参数
参数含义可以查看[vllm文档](https://docs.vllm.ai/en/latest/serving/engine_args.html)。

- 🔥vllm_gpu_memory_utilization: GPU内存比例，取值范围为0到1。默认值`0.9`。
  - 注意：该参数在"ms-swift<3.7"的参数名为`gpu_memory_utilization`。下面的`vllm_`参数同理。若出现参数不匹配问题，请查看[ms-swift3.6文档](https://swift.readthedocs.io/zh-cn/v3.6/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html#vllm)。
- 🔥vllm_tensor_parallel_size: tp并行数，默认为`1`。
- vllm_pipeline_parallel_size: pp并行数，默认为`1`。
- vllm_data_parallel_size: dp并行数，默认为`1`，在`swift deploy/rollout`命令中生效。
  - 若在`swift infer`中，使用`NPROC_PER_NODE`来设置dp并行数。参考这里的[例子](https://github.com/modelscope/ms-swift/blob/main/examples/infer/vllm/mllm_ddp.sh)。
- vllm_enable_expert_parallel: 开启专家并行，默认为False。
- vllm_max_num_seqs: 单次迭代中处理的最大序列数，默认为`256`。
- 🔥vllm_max_model_len: 模型支持的最大长度。默认为`None`，即从config.json中读取。
- vllm_disable_custom_all_reduce: 禁用自定义的 all-reduce 内核，回退到 NCCL。为了稳定性，默认为`True`。
- vllm_enforce_eager: vllm使用pytorch eager模式还是建立cuda graph，默认为`False`。设置为True可以节约显存，但会影响效率。
- vllm_mm_processor_cache_gb: 多模态处理器缓存大小（GiB），用于缓存已处理的多模态输入（如图像、视频）避免重复处理。默认为`4`。设置为`0`可禁用缓存但会降低性能（不推荐）。仅对多模态模型生效。
- vllm_speculative_config: 推测解码配置，传入json字符串。默认为None。
- vllm_disable_cascade_attn: 是否强制关闭V1引擎的cascade attention实现以防止潜在数值误差，默认为False，由vLLM内部逻辑决定是否使用。
- 🔥vllm_limit_mm_per_prompt: 控制vllm使用多图，默认为`None`。例如传入`--vllm_limit_mm_per_prompt '{"image": 5, "video": 2}'`。
- vllm_max_lora_rank: 默认为`16`。vllm对于lora支持的参数。
- vllm_quantization: vllm可以在内部量化模型，参数支持的值详见[这里](https://docs.vllm.ai/en/latest/serving/engine_args.html)。
- 🔥vllm_enable_prefix_caching: 开启vllm的自动前缀缓存，节约重复查询前缀的处理时间，加快推理效率。默认为`None`，跟随vLLM行为。
  - 该参数在"ms-swift<3.9.1"的默认值为`False`。
- vllm_use_async_engine: vLLM backend下是否使用async engine。默认为None，会根据场景自动设置：encode任务（embedding、seq_cls、reranker、generative_reranker）默认为True，部署场景（swift deploy）默认为True，其他情况默认为False。注意：encode任务需使用async engine。
- vllm_reasoning_parser: 推理解析器类型，用于思考模型的思维链内容解析。默认为`None`。仅用于 `swift deploy` 命令。可选的种类参考[vLLM文档](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#streaming-chat-completions)。
- vllm_engine_kwargs: vllm的额外参数，格式为json字符串。默认为None。

### SGLang参数
参数含义可以查看[sglang文档](https://docs.sglang.ai/backend/server_arguments.html)。

- 🔥sglang_tp_size: tp数。默认为1。
- sglang_pp_size: pp数。默认为1。
- sglang_dp_size: dp数。默认为1。
- sglang_ep_size: ep数。默认为1。
- sglang_enable_ep_moe: 是否启用ep moe。默认为False。该参数已在最新sglang中移除。
- sglang_mem_fraction_static: 用于静态分配模型权重和KV缓存内存池的GPU内存比例。如果你遇到GPU内存不足错误，可以尝试降低该值。默认为None。
- sglang_context_length: 模型的最大上下文长度。默认为 None，将使用模型的`config.json`中的值。
- sglang_disable_cuda_graph: 禁用CUDA图。默认为False。
- sglang_quantization: 量化方法。默认为None。
- sglang_kv_cache_dtype: 用于k/v缓存存储的数据类型。'auto'表示将使用模型的数据类型。'fp8_e5m2'和'fp8_e4m3'适用于CUDA 11.8及以上版本。默认为'auto'。
- sglang_enable_dp_attention: 为注意力机制启用数据并行，为前馈网络（FFN）启用张量并行。数据并行的规模（dp size）应等于张量并行的规模（tp size）。目前支持DeepSeek-V2/3以及Qwen2/3 MoE模型。默认为False。
- sglang_disable_custom_all_reduce: 禁用自定义的 all-reduce 内核，回退到 NCCL。为了稳定性，默认为True。
- sglang_speculative_algorithm: 推测算法，可选值：None、"EAGLE"、"EAGLE3"、"NEXTN"、"STANDALONE"、"NGRAM"。默认为None。
- sglang_speculative_num_steps: 在推测解码中从草稿模型采样的步数。默认值为None。
- sglang_speculative_eagle_topk: 在 EAGLE2 算法中每步从草稿模型采样的 token 数量。默认值为None。
- sglang_speculative_num_draft_tokens: 在推测解码中从草稿模型采样的 token 数量。默认值为None。

### LMDeploy参数
参数含义可以查看[lmdeploy文档](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig)。

- 🔥lmdeploy_tp: tensor并行度。默认为`1`。
- lmdeploy_session_len: 最大会话长度。默认为`None`。
- lmdeploy_cache_max_entry_count: k/v缓存占用的GPU内存百分比。默认为`0.8`。
- lmdeploy_quant_policy: 默认为0。当需要将k/v量化为4或8位时，分别将其设置为4或8。
- lmdeploy_vision_batch_size: 传入VisionConfig的max_batch_size参数。默认为`1`。

### 合并参数

- 🔥merge_lora: 是否合并lora，本参数支持lora、llamapro、longlora，默认为False。例子参数[这里](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh)。
- safe_serialization: 是否存储为safetensors，默认为True。
- max_shard_size: 单存储文件最大大小，默认'5GB'。


## 集成参数

### 训练参数
训练参数除包含[基本参数](#基本参数)、[Seq2SeqTrainer参数](#Seq2SeqTrainer参数)、[tuner参数](#tuner参数)外，还包含下面的部分:

- add_version: 在output_dir上额外增加目录`'<版本号>-<时间戳>'`防止权重覆盖，默认为True。
- check_model: 检查本地模型文件有损坏或修改并给出提示，默认为True。**如果是断网环境，请设置为False**。
- 🔥create_checkpoint_symlink: 额外创建checkpoint软链接，方便书写自动化训练脚本。best_model和last_model的软链接路径分别为f'{output_dir}/best'和f'{output_dir}/last'。
- 🔥packing: 使用`padding_free`的方式将不同长度的数据样本打包成**近似**统一长度的样本（packing能保证不对完整的序列进行切分），实现训练时各节点与进程的负载均衡（避免长文本拖慢短文本的训练速度），从而提高GPU利用率，保持显存占用稳定。当使用 `--attn_impl flash_attn` 时，可确保packed样本内的不同序列之间相互独立，互不可见。该参数默认为`False`，目前支持 CPT/SFT/DPO/KTO/GKD。注意：**packing会导致数据集样本数减少，请自行调节梯度累加数和学习率**。
  - "ms-swift>=3.12"新支持了embedding/reranker/seq_cls任务的packing。
- packing_length: packing的长度。默认为None，设置为max_length。
- packing_num_proc: packing的进程数，默认为1。需要注意的是，不同的`packing_num_proc`，最终形成的packed数据集是不同的。（该参数在流式packing时不生效）。通常不需要修改该值，packing速度远快于tokenize速度。
- lazy_tokenize: 是否使用lazy_tokenize。若该参数设置为False，则在训练之前对所有的数据集样本进行tokenize（多模态模型则包括从磁盘中读取图片）。该参数默认为None，在LLM训练中默认为False，而MLLM训练默认为True，节约内存。
  - 注意：若你要进行图像的数据增强，你需要将lazy_tokenize（或streaming）设置为True，并修改Template类中的encode方法。
- use_logits_to_keep: 通过在`forward`中根据labels传入logits_to_keep，减少无效logits的计算与存储，从而减少显存占用并加快训练速度。默认为None，进行自动选择。
- acc_strategy: 训练和验证时计算acc的策略。可选为`seq`和`token`级别的acc，默认为`token`。
- max_new_tokens: 覆盖生成参数。predict_with_generate=True时的最大生成token数量，默认64。
- temperature: 覆盖生成参数。predict_with_generate=True时的temperature，默认0。
- optimizer: 使用的optimizers插件（优先级高于`--optim`），默认为None。可选optimizers参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/optimizers/mapping.py)。
- loss_type: 自定义的loss_type名称。默认为None，使用模型自带损失函数。可选loss参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py)。
- eval_metric: 自定义eval metric名称。默认为None。可选eval_metric参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/eval_metric/mapping.py)。
  - 关于默认值：当`task_type`为'causal_lm', 且`predict_with_generate=True`的情况下默认设置为'nlg'。`task_type` 为'embedding'，根据loss_type，默认值为'infonce' 或 'paired'。`task_type`为'reranker/generative_reranker'，默认值为'reranker'。
- callbacks: 自定义trainer callback，默认为`[]`。可选callbacks参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/callbacks/mapping.py)。例如：通过在`callbacks`中添加`deepspeed_elastic`（可选`graceful_exit`）可以来启用弹性训练。参考[Elastic示例](../BestPractices/Elastic.md)
- early_stop_interval: 早停的间隔，会检验best_metric在early_stop_interval个周期内（基于`save_steps`, 建议`eval_steps`和`save_steps`设为同值）没有提升时终止训练。具体代码在[early_stop.py](https://github.com/modelscope/ms-swift/blob/main/swift/callbacks/early_stop.py)中。同时，如果有较为复杂的早停需求，直接覆盖callback.py中的已有实现即可。设置该参数时，自动加入`early_stop`的trainer callback。
- eval_use_evalscope: 是否使用evalscope进行训练时评测，需要设置该参数来开启评测，具体使用参考[示例](../Instruction/Evaluation.md#训练中评测)。
- eval_dataset: 评测数据集，可设置多个数据集，用空格分割。
- eval_dataset_args: 评测数据集参数，json格式，可设置多个数据集的参数。
- eval_limit: 评测数据集采样数。
- eval_generation_config: 评测时模型推理配置，json格式，默认为`{'max_tokens': 512}`。
- use_flash_ckpt: 是否启用[DLRover Flash Checkpoint](https://github.com/intelligent-machine-learning/dlrover)的flash checkpoint。默认为`false`，启用后，权重会先保存至共享内存，之后异步持久化；建议搭配`PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` 一起使用，避免训练过程CUDA OOM。

#### SWANLAB

- swanlab_token: SwanLab的api-key。你也可以使用`SWANLAB_API_KEY`环境变量指定。
- swanlab_project: swanlab的project，可以在页面中预先创建[https://swanlab.cn/space/~](https://swanlab.cn/space/~)或自动创建，默认为"ms-swift"。
- swanlab_workspace: 默认为None，会使用api-key对应的username。
- swanlab_exp_name: 实验名，可以为空，为空时默认传入--output_dir的值。
- swanlab_notification_method: 在训练完成/发生错误时，swanlab的通知方式，具体参考[这里](https://docs.swanlab.cn/plugin/notification-dingtalk.html)。支持'dingtalk'、'lark'、'email'、'discord'、'wxwork'、'slack'。
- swanlab_webhook_url: 默认为None。swanlab的`swanlab_notification_method`对应的 webhook url。
- swanlab_secret: 默认为None。swanlab的`swanlab_notification_method`对应的 secret。
- swanlab_mode: 可选cloud和local，云模式或者本地模式。


### RLHF参数
RLHF参数继承于[训练参数](#训练参数)。

- 🔥rlhf_type: 人类对齐算法类型，支持'dpo'、'orpo'、'simpo'、'kto'、'cpo'、'rm'、'ppo'、'grpo'和'gkd'。默认为'dpo'。
- ref_model: 采用dpo、kto、ppo、grpo算法且使用全参数训练时需要传入。默认为None，设置为`--model`。
- ref_adapters: 默认为`[]`。若你要使用SFT产生的LoRA权重进行DPO/KTO/GRPO，请使用"ms-swift>=3.8"，并在训练时设置`--adapters sft_ckpt --ref_adapters sft_ckpt`。若是此场景的断点续训，则设置`--resume_from_checkpoint rlhf_ckpt --ref_adapters sft_ckpt`。
- ref_model_type: 同model_type。默认为None。
- ref_model_revision: 同model_revision。默认为None。
- 🔥beta: 控制与参考模型偏差程度的参数。beta值越高，表示与参考模型的偏差越小。默认为`None`，使用不同rlhf算法的默认值不同，其中`simpo`算法默认为`2.`，GRPO默认为`0.04`，GKD默认为0.5，其他算法默认为`0.1`。具体参考[文档](./RLHF.md)。
- label_smoothing: 是否使用DPO smoothing，默认值为`0`。
- max_completion_length: GRPO/PPO/GKD算法中的最大生成长度，默认为512。
- 🔥rpo_alpha: 来自[RPO 论文](https://arxiv.org/abs/2404.19733)中的参数，用于控制损失函数中NLL项的权重（即SFT损失），`loss = dpo_loss + rpo_alpha * sft_loss`，论文中推荐设置为`1.`。默认为`None`，即默认不引入sft_loss。
  - **注意**：在"ms-swift<3.8"，其默认值为`1.`。在"ms-swift>=3.8"该默认值修改为`None`。
- ld_alpha: 来自[LD-DPO 论文](https://arxiv.org/abs/2409.06411)，对超出公共前缀部分的logps加权 $\alpha$ 抑制长度偏好。
- discopop_tau: 来自 [DiscoPOP 论文](https://arxiv.org/abs/2406.08414)的温度参数 $\tau$ ，用于缩放 log-ratio。默认值0.05。在 loss_type 为 discopop 时生效。
- loss_type: 损失类型。默认为None，使用不同的rlhf算法，其默认值不同。
  - DPO: 可选项参考[文档](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions)，支持传入多个值实现混合训练([MPO](https://arxiv.org/abs/2411.10442)), 传入多个值时需要设置参数 loss_weights。默认为`sigmoid`。
  - GRPO: 参考[GRPO参数](#grpo参数)。
- loss_weights：在 DPO 训练中设置多个 loss_type 时，用于指定各个损失项的权重。
- cpo_alpha: CPO/SimPO loss 中 nll loss的系数, 默认为`1.`。
- simpo_gamma: SimPO算法中的reward margin项，论文建议设置为0.5-1.5，默认为`1.`。
- desirable_weight: KTO算法中用于抵消 desirable 和 undesirable 数量不均衡的影响，对 desirable 损失按该系数进行加权，默认为`1.`。
- undesirable_weight: KTO算法中用于抵消 desirable 和 undesirable 数量不均衡的影响，对 undesirable 损失按该系数进行加权，默认为`1.`。
- center_rewards_coefficient: 用于RM训练。用于激励奖励模型输出均值为零的奖励的系数，具体查看这篇[论文](https://huggingface.co/papers/2312.09244)。推荐值：0.01。
- loss_scale: 覆盖模板参数。rlhf训练时，默认为'last_round'。
- temperature: 默认为0.9，该参数将在PPO、GRPO、GKD中使用。

#### GKD参数
- lmbda: 默认为0.5。该参数在GKD中使用。控制学生数据比例的 lambda 参数（即策略内学生生成输出所占的比例）。若lmbda为0，则不使用学生生成数据。
- sft_alpha: 默认为0。控制GKD中加入sft_loss的权重。最后的loss为`gkd_loss + sft_alpha * sft_loss`。
- seq_kd: 默认为False。该参数在GKD中使用。控制是否执行序列级知识蒸馏（Sequence-Level KD）的 seq_kd 参数（可视为对教师模型生成输出的监督式微调）。
  - 注意：你可以提前对数据集内容使用teacher模型进行推理（使用vllm/sglang/lmdeploy等推理引擎加速），并在训练时将`seq_kd`设置为False。或者将`seq_kd`设置为True，在训练时使用teacher模型生成序列（能保证多个epoch生成数据的不同，但效率较慢）。
- offload_teacher_model: 卸载教师模型以节约显存，只在采样/计算logps时加载，默认为False。
- truncation_strategy：用于处理输入长度超过 max_length 的样本，支持 delete 和 left 两种策略，分别表示删除该样本和从左侧裁剪。默认值为 left。若使用 delete 策略，被删除的超长样本或编码失败的样本将在原数据集中通过重采样进行替换。
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb/swanlab` 使用。默认为False。
  - 提示：若没有设置`--report_to wandb/swanlab`，则会在checkpoint中创建`completions.jsonl`来存储生成内容。
  - 仅记录 vLLM 采样结果。

#### Reward/Teacher模型参数
reward模型参数将在PPO、GRPO中使用。

- reward_model: 默认为None。
- reward_adapters: 默认为`[]`。
- reward_model_type: 默认为None。
- reward_model_revision: 默认为None。
- teacher_model: 默认为None。rlhf_type为'gkd'时需传入此参数。
- teacher_adapters: 默认为`[]`。
- teacher_model_type: 默认为None。
- teacher_model_revision: 默认为None。
- teacher_deepspeed: 同 deepspeed 参数，控制 teacher model 的 deepspeed 配置，默认使用训练模型的 deepspeed 配置。

#### PPO参数

以下参数含义可以参考[这里](https://huggingface.co/docs/trl/main/ppo_trainer)。
- num_ppo_epochs: 默认为4。
- whiten_rewards: 默认为False。
- kl_coef: 默认为0.05。
- cliprange: 默认为0.2。
- vf_coef: 默认为0.1。
- cliprange_value: 默认为0.2。
- gamma: 默认为1.0。
- lam: 默认为0.95。
- num_mini_batches: 默认为1。
- local_rollout_forward_batch_size: 默认为64。
- num_sample_generations: 默认为10。
- missing_eos_penalty: 默认为None。


#### GRPO参数
- beta: KL正则系数，默认为0.04，设置为0时不加载ref model。
- per_device_train_batch_size: 每个设备训练批量大小，在GRPO中，指 completion 的批次大小。
- per_device_eval_batch_size: 每个设备评估批量大小，在GRPO中，指 completion 的批次大小。
- generation_batch_size: 采样completion批量大小，需要是 num_processes * per_device_train_batch_size 的倍数，默认等于 per_device_batch_size * gradient_accumulation_steps * num_processes
- steps_per_generation: 每轮生成的优化步数，默认等于gradient_accumulation_steps。与generation_batch_size 只能同时设置一个
- num_generations: 每个prompt采样的数量，论文中的G值，采样批量大小(generation_batch_size 或 steps_per_generation × per_device_batch_size × num_processes) 必须能被 num_generations 整除。默认为 8。
- num_generations_eval: 评估阶段每个prompt采样的数量。允许在评估时使用较少的生成数量以节省计算资源。如果为 None，则使用 num_generations 的值。默认为 None。
- ds3_gather_for_generation: 该参数适用于DeepSpeed ZeRO-3。如果启用，策略模型权重将被收集用于生成，从而提高生成速度。然而，禁用此选项允许训练超出单个GPU VRAM的模型，尽管生成速度会变慢。禁用此选项与vLLM生成不兼容。默认为True。
- reward_funcs: GRPO算法奖励函数，可选项为`accuracy`、`format`、`cosine`、`repetition`和`soft_overlong`，见swift/rewards/orm.py。你也可以在plugin中自定义自己的奖励函数。默认为`[]`。
- reward_weights: 每个奖励函数的权重。必须与奖励函数和奖励模型的总数量匹配。如果为 None，则所有奖励的权重都相等，为`1.0`。
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置。
- reward_model_plugin: 奖励模型逻辑，默认为orm逻辑, 详细见[自定义奖励模型](./GRPO/DeveloperGuide/reward_model.md#自定义奖励模型)。
- dataset_shuffle: 是否对dataset进行随机操作，默认为True。
- truncation_strategy：用于处理输入长度超过 max_length 的样本，支持 delete 和 left 两种策略，分别表示删除该样本和从左侧裁剪。默认值为 left。若使用 delete 策略，被删除的超长样本或编码失败的样本将在原数据集中通过重采样进行替换。
- loss_type: loss 归一化的类型，可选项为['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo'], 默认为'grpo', 具体参考[文档](./GRPO/DeveloperGuide/loss_types.md)
- log_completions: 是否记录训练中的模型生成内容，搭配 `--report_to wandb/swanlab` 使用。默认为False。
  - 提示：若没有设置`--report_to wandb/swanlab`，则会在checkpoint中创建`completions.jsonl`来存储生成内容。
- use_vllm: 是否使用 vLLM 作为 GRPO 生成的 infer_backend，默认为False。
- vllm_mode: vLLM 集成模式，可选项为 `server` 和 `colocate`。server 模式使用 `swift rollout` 拉起的 vLLM 服务器进行采样，colocate 模式在程序内部署 vLLM。使用server端时，
- vllm_mode server 参数
  - vllm_server_host: vLLM server host地址，默认为None。
  - vllm_server_port: vLLM server 服务端口，默认为8000。
  - vllm_server_base_url: vLLM server的Base URL(比如 http://local_host:8000), 默认为None。设置后，忽略host和port设置。
  - vllm_server_group_port: vllm server 内部通信端口，除非端口被占用，一般无需设置，默认为51216。
  - vllm_server_timeout: 连接vLLM server的超时时间，默认为 240s。
  - vllm_server_pass_dataset: 透传额外的数据集信息到vLLM server，用于多轮训练。
  - async_generate: 异步rollout以提高训练速度，注意开启时采样会使用上一轮更新的模型进行采样，不支持多轮场景。默认`false`.
  - enable_flattened_weight_sync: 是否使用 flattened tensor 进行权重同步。启用后会将多个参数打包为单个连续 tensor 进行传输，可提升同步效率，在 Server Mode 下生效，默认为 True。
  - SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE: 环境变量，用于控制flattened tensor 权重同步时的传输桶大小（bucket size），适用于 Server Mode 下的全参数训练，单位为 MB，默认值为 512 MB。
- vllm_mode colocate 参数（更多参数支持参考[vLLM参数](#vLLM参数)。）
  - vllm_gpu_memory_utilization: vllm透传参数，默认为0.9。
  - vllm_max_model_len: vllm透传参数，默认为None。
  - vllm_enforce_eager: vllm透传参数，默认为False。
  - vllm_limit_mm_per_prompt: vllm透传参数，默认为None。
  - vllm_enable_prefix_caching: vllm透传参数，默认为True。
  - vllm_tensor_parallel_size: tp并行数，默认为`1`。
  - vllm_enable_lora: 支持vLLM Engine 加载 LoRA adapter，默认为False。用于加速LoRA训练的权重同步，具体参考[文档](./GRPO/GetStarted/GRPO.md#权重同步加速)。
  - sleep_level: 训练时释放 vLLM 显存，可选项为[0, 1, 2], 默认为0，不释放。
  - offload_optimizer: 是否在vLLM推理时offload optimizer参数，默认为False。
  - offload_model: 是否在vLLM推理时 offload 模型，默认为False。
  - completion_length_limit_scope: 在多轮对话中，`max_completion_length` 的限制范围。
  `total`限制所有对话轮次的总输出长度不超过`max_completion_length`, `per_round`限制每一轮的输出长度。
- num_iterations: 每条数据的更新次数，[GRPO论文](https://arxiv.org/abs/2402.03300)中的 $\mu$ 值，默认为1。
- epsilon: clip 系数，默认为0.2。
- epsilon_high: upper clip 系数，默认为None，设置后与epsilon共同构成[epsilon, epsilon_high]裁剪范围。
- tau_pos: [SAPO](https://arxiv.org/abs/2511.20347)算法中正优势的温度参数，控制软门控函数的锐度。较大值使门控更锐利（接近硬裁剪），较小值使门控更平滑。默认为1.0。
- tau_neg: SAPO算法中负优势的温度参数，控制软门控函数的锐度。通常设置`tau_neg > tau_pos`以对负优势施加更强约束。默认为1.05。
- dynamic_sample：筛除group内奖励标准差为0的数据，额外采样新数据，默认为False。
- max_resample_times：dynamic_sample设置下限制重采样次数，默认3次。
- overlong_filter：跳过超长截断的样本，不参与loss计算，默认为False。
- delta: [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)中双侧 GRPO 上界裁剪值。若设置，建议大于 1 + epsilon。默认为None。
- importance_sampling_level: 控制重要性采样比计算，可选项为 `token` 和 `sequence`，`token` 模式下保留原始的每个 token 的对数概率比，`sequence` 模式下则会对序列中所有有效 token 的对数概率比进行平均。[GSPO论文](https://arxiv.org/abs/2507.18071)中使用sequence级别计算来稳定训练，默认为`token`。
- advantage_estimator: 优势计算函数，默认为 `grpo`，即计算组内相对优势，可选项为 `grpo`、[`rloo`](./GRPO/AdvancedResearch/RLOO.md)、[`reinforce_plus_plus`](./GRPO/AdvancedResearch/REINFORCEPP.md)。
- kl_in_reward: 控制 KL 散度正则项的处理位置；`false`表示作为损失函数的独立正则项，`true`表示将 KL 直接并入奖励（从奖励中扣除）。默认情况与advantage_estimator绑定，`grpo`下默认为`false`，`rloo` 和 `reinforce_plus_plus` 下默认为 `true`。
- scale_rewards：指定奖励的缩放策略。可选值包括 `group`（按组内标准差缩放）、`batch`（按整个批次的标准差缩放）、`none`（不进行缩放）、`gdpo`（对每个奖励函数分别进行组内归一化后加权聚合，参考 [GDPO 论文](https://arxiv.org/abs/2601.05242)）。在 ms-swift < 3.10 版本中，该参数为布尔类型，`true` 对应 `group`，`false` 对应 `none`。默认值与 `advantage_estimator` 绑定：`grpo` 对应 `group`，`rloo` 对应 `none`，`reinforce_plus_plus` 对应 `batch`。
  - 注意：`gdpo` 模式不支持 `kl_in_reward=True`，若同时设置会自动将 `kl_in_reward` 设为 `False`。
  - GDPO 适用于多奖励优化场景：当使用多个奖励函数时，GDPO 会对每个奖励函数分别在组内进行标准化（减均值、除标准差），然后使用 `reward_weights` 进行加权求和，最后再进行批次级别的标准化。这种方式可以更好地保留各个奖励的相对差异，避免不同奖励组合坍塌成相同的 advantage 值。
- sync_ref_model: 是否定期同步ref_model，默认为False。
  - ref_model_mixup_alpha: 控制在更新过程中model和先前ref_model之间的混合。更新公式为 $π_{ref} = α * π_θ + (1 - α) * π_{ref_{prev}}$。默认为0.6。
  - ref_model_sync_steps：同步频率，默认为512。
- move_model_batches: 在模型向vLLM等快速推理框架移动参数时，将layers分为多少个batch. 默认为None, 代表整个模型不进行拆分，否则拆分为move_model_batches+1(非layer参数)+1(多模态部分参数)个。
- multi_turn_scheduler: 多轮GRPO参数, 传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现。
- max_turns: 多轮GRPO的轮数上限。默认为None，不做限制。
- top_entropy_quantile: 仅对熵值处于前指定分位的 token 参与损失计算，默认为1.0，即不过滤低熵 token，具体参考[文档](./GRPO/AdvancedResearch/entropy_mask.md)
- log_entropy: 记录训练中的熵值变化动态，默认为False，具体参考[文档](./GRPO/GetStarted/GRPO.md#logged-metrics)
- rollout_importance_sampling_mode: 训推不一致校正模式，可选项为 `token_truncate`、`token_mask`、`sequence_truncate`、`sequence_mask`。默认为None，不启用校正。具体参考[文档](./GRPO/AdvancedResearch/training_inference_mismatch.md)
- rollout_importance_sampling_threshold: 重要性采样权重的阈值，用于截断或屏蔽极端权重。默认为2.0。
- log_rollout_offpolicy_metrics: 当 `rollout_importance_sampling_mode` 未设置时，是否记录训推不一致诊断指标（KL、PPL、χ²等）。当设置了 `rollout_importance_sampling_mode` 时，指标会自动记录。默认为False。
- off_policy_sequence_mask_delta: Off-Policy Sequence Masking 阈值，来自 [DeepSeek-V3.2 论文](https://arxiv.org/abs/2512.02556)。当设置此值时，会计算每个序列的 `mean(old_policy_logps - policy_logps)`，若该值大于阈值且该序列的优势为负，则 mask 掉该序列不参与损失计算。具体参考[文档](./GRPO/AdvancedResearch/training_inference_mismatch.md#off-policy-sequence-masking)

##### 奖励函数参数
内置的奖励函数参考[文档](./GRPO/DeveloperGuide/reward_function.md)
cosine 奖励参数
- cosine_min_len_value_wrong：cosine 奖励函数参数，生成错误答案时，最小长度对应的奖励值。默认值为-0.5。
- cosine_max_len_value_wrong：生成错误答案时，最大长度对应的奖励值。默认值为0.0。
- cosine_min_len_value_correct：生成正确答案时，最小长度对应的奖励值。默认值为1.0。
- cosine_max_len_value_correct：生成正确答案时，最大长度对应的奖励值。默认值为0.5。
- cosine_max_len：生成文本的最大长度限制。默认等于 max_completion_length。

repetition 奖励参数
- repetition_n_grams：用于检测重复的 n-gram 大小。默认值为3。
- repetition_max_penalty：最大惩罚值，用于控制惩罚的强度。默认值为-1.0。

soft overlong 奖励参数
- soft_max_length: 论文中的L_max，模型的最大生成长度，默认等于max_completion_length。
- soft_cache_length: 论文中的L_cache，控制长度惩罚区间，区间为[soft_max_length-soft_cache_length, soft_max_length]。

### 推理参数

推理参数除包含[基本参数](#基本参数)、[合并参数](#合并参数)、[vLLM参数](#vllm参数)、[LMDeploy参数](#LMDeploy参数)外，还包含下面的部分：

- 🔥infer_backend: 推理加速后端，支持'transformers'、'vllm'、'sglang'、'lmdeploy'四种推理引擎。默认为'transformers'。
  - 注意：这四种引擎使用的都是swift的template，使用`--template_backend`控制。
- 🔥max_batch_size: 指定infer_backend为'transformers'时生效，用于批量推理，默认为1。若设置为-1，则不受限制。
- 🔥result_path: 推理结果存储路径（jsonl），默认为None，保存在checkpoint目录（含args.json文件）或者'./result'目录，最终存储路径会在命令行中打印。
  - 注意：若已存在`result_path`文件，则会进行追加写入。
- write_batch_size: 结果写入`result_path`的batch_size。默认为1000。若设置为-1，则不受限制。
- metric: 对推理的结果进行评估，目前支持'acc'和'rouge'。默认为None，即不进行评估。
- val_dataset_sample: 推理数据集采样数，默认为None。
- reranker_use_activation: 在reranker推理时，是否在score之后使用sigmoid，默认为True。


### 部署参数

部署参数继承于[推理参数](#推理参数)。

- host: 服务host，默认为'0.0.0.0'。
- port: 端口号，默认为8000。
- api_key: 访问需要使用的api_key，默认为None。
- owned_by: 默认为`swift`。
- 🔥served_model_name: 提供服务的模型名称，默认使用model的后缀。
- verbose: 打印详细日志，默认为True。
  - 注意：在`swift app`或者`swift eval`时，默认为False。
- log_interval: tokens/s统计值打印间隔，默认20秒。设置为-1则不打印。
- max_logprobs: 最多返回客户端的logprobs数量，默认为20。

### Rollout参数
Rollout参数继承于[部署参数](#部署参数)
- multi_turn_scheduler: 多轮GRPO训练规划器，传入对应的plugin名称, 同时在plugin/multi_turn.py中添加好对应的实现。默认为None，具体参考[文档](./GRPO/DeveloperGuide/multi_turn.md)。
- max_turns: 多轮GRPO训练下的最大轮数，默认为None，即不做约束。
- vllm_enable_lora: 支持vLLM Engine 加载 LoRA adapter，默认为False。用于加速LoRA训练的权重同步，具体参考[文档](./GRPO/GetStarted/GRPO.md#权重同步加速)。
- vllm_max_lora_rank: vLLM Engine LoRA参数，需大于等于训练的lora_rank，建议等于。默认为16。

### Web-UI参数
- server_name: web-ui的host，默认为'0.0.0.0'。
- server_port: web-ui的port，默认为7860。
- share: 默认为False。
- lang: web-ui的语言，可选为'zh', 'en'。默认为'zh'。


### App参数

App参数继承于[部署参数](#部署参数), [Web-UI参数](#Web-UI参数)。
- base_url: 模型部署的base_url，例如`http://localhost:8000/v1`。默认为`None`，使用本地部署。
- studio_title: studio的标题。默认为None，设置为模型名。
- is_multimodal: 是否启动多模态版本的app。默认为None，自动根据model判断，若无法判断，设置为False。
- lang: 覆盖Web-UI参数，默认为'en'。

### 评测参数

评测参数继承于[部署参数](#部署参数)。

- 🔥eval_backend: 评测后端，默认为'Native'，也可以指定为'OpenCompass'或'VLMEvalKit'。
- 🔥eval_dataset: 评测数据集，请查看[评测文档](./Evaluation.md)。
- eval_limit: 每个评测集的采样数，默认为None。
- eval_output_dir: 评测存储结果的文件夹，默认为'eval_output'。
- temperature: 覆盖生成参数，默认为0。
- eval_num_proc: 评测时客户端最大并发数，默认为16。
- eval_url: 评测url，例如`http://localhost:8000/v1`。例子可以查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/eval/eval_url)。默认为None，采用本地部署评估。
- eval_generation_config: 评测时模型推理配置，需传入json字符串格式，例如：`'{"max_new_tokens": 512}'`；默认为None。
- extra_eval_args: 额外评测参数，需传入json字符串格式，默认为空。仅对Native评测有效，更多参数说明请查看[这里](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html)
- local_dataset: 部分评测集，如`CMB`无法直接运行，需要下载额外数据包才可以使用。设置本参数为`true`可以自动下载全量数据包，并在当前目录下创建`data`文件夹并开始评测。数据包仅会下载一次，后续会使用缓存。该参数默认为`false`。
  - 注意：默认评测会使用`~/.cache/opencompass`下的数据集，在指定本参数后会直接使用当前目录下的data文件夹。

### 导出参数

导出参数除包含[基本参数](#基本参数)和[合并参数](#合并参数)外，还包含下面的部分:

- 🔥output_dir: 导出结果存储路径。默认为None，会自动设置合适后缀的路径。
- exist_ok: 如果output_dir存在，不抛出异常，进行覆盖。默认为False。
- 🔥quant_method: 可选为'gptq'、'awq'、'bnb'和'fp8'，默认为None。例子参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize)。
- quant_n_samples: gptq/awq的校验集采样数，默认为256。
- quant_batch_size: 量化batch_size，默认为1。
- group_size: 量化group大小，默认为128。
- to_cached_dataset: 提前对数据集进行tokenize并导出，默认为False。例子参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/cached_dataset)。更多介绍请查看`cached_dataset`。
  - 提示：你可以通过`--split_dataset_ratio`或者`--val_dataset`指定验证集内容。
- template_mode: 用于支持对`swift rlhf`训练的`cached_dataset`功能。该参数只在`--to_cached_dataset true`时生效。可选项包括: 'train'、'rlhf'和'kto'。其中`swift pt/sft`使用'train'，`swift rlhf --rlhf_type kto`使用'kto'，其他rlhf算法使用'rlhf'。注意：当前'gkd', 'ppo', 'grpo'算法不支持`cached_dataset`功能。默认为'train'。
- to_ollama: 产生ollama所需的Modelfile文件。默认为False。
- 🔥to_mcore: HF格式权重转成Megatron格式。默认为False。
- to_hf: Megatron格式权重转成HF格式。默认为False。
- mcore_model: mcore格式模型路径。默认为None。
- mcore_adapters: mcore格式模型的adapter路径列表，默认为空列表。
- thread_count: `--to_mcore true`时的模型切片数。默认为None，根据模型大小自动设置，使得最大分片小于10GB。
- 🔥offload_bridge: Megatron导出的用于vLLM更新HF格式权重使用CPU主存存放，以降低 GPU 显存占用。默认为 False。
- 🔥test_convert_precision: 测试HF和Megatron格式权重转换的精度误差。默认为False。
- test_convert_dtype: 转换精度测试使用的dtype，默认为'float32'。
- 🔥push_to_hub: 是否推送hub，默认为False。例子参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/export/push_to_hub.sh)。
- hub_model_id: 推送的model_id，默认为None。
- hub_private_repo: 是否是private repo，默认为False。
- commit_message: 提交信息，默认为'update files'。

### 采样参数

- prm_model: 过程奖励模型的类型，可以是模型id（以'transformers'方式拉起），或者plugin中定义的prm key（自定义推理过程）。
- orm_model: 结果奖励模型的类型，通常是通配符或测试用例等，一般定义在plugin中。
- sampler_type：采样类型，目前支持 sample, distill
- sampler_engine：支持`transformers`, `lmdeploy`, `vllm`, `client`, `no`，默认为`transformers`，采样模型的推理引擎。
- output_dir：输出目录，默认为`sample_output`。
- output_file：输出文件名称，默认为`None`使用时间戳作为文件名。传入时不需要传入目录，仅支持jsonl格式。
- override_exist_file：如`output_file`存在，是否覆盖。
- num_sampling_batch_size：每次采样的batch_size。
- num_sampling_batches：共采样多少batch。
- n_best_to_keep：返回多少最佳sequences。
- data_range：本采样处理数据集的分片。传入格式为`2 3`，代表数据集分为3份处理（这意味着通常有三个`swift sample`在并行处理），本实例正在处理第3个分片。
- temperature：在这里默认为1.0。
- prm_threshold：PRM阈值，低于该阈值的结果会被过滤掉，默认值为`0`。
- easy_query_threshold：单个query的所有采样中，ORM评估如果正确，大于该比例的query会被丢弃，防止过于简单的query出现在结果中，默认为`None`，代表不过滤。
- engine_kwargs：传入sampler_engine的额外参数，以json string传入，例如`{"cache_max_entry_count":0.7}`。
- num_return_sequences：采样返回的原始sequence数量。默认为64，本参数对`sample`采样有效。
- cache_files：为避免同时加载prm和generator造成显存OOM，可以分两步进行采样，第一步将prm和orm置为`None`，则所有结果都会输出到文件中，第二次运行采样将sampler_engine置为`no`并传入`--cache_files`为上次采样的输出文件，则会使用上次输出的结果进行prm和orm评估并输出最终结果。
  - 注意：使用cache_files时，`--dataset`仍然需要传入，这是因为cache_files的id是由原始数据计算的md5，需要把两部分信息结合使用。

## 特定模型参数
除了以上参数外，有些模型还支持额外的具体模型参数。这些参数含义通常可以在对应模型官方repo或者其推理代码中找到相应含义。**ms-swift引入这些参数以确保训练的模型与官方推理代码效果对齐**。
- 特定模型参数可以通过`--model_kwargs`或者环境变量进行设置，例如: `--model_kwargs '{"fps_max_frames": 12}'`或者`FPS_MAX_FRAMES=12`。
- 注意：若你在训练时指定了特定模型参数，请在推理时也设置对应的参数，这可以提高训练效果。

### qwen2_vl, qvq, qwen2_5_vl, mimo_vl, keye_vl, keye_vl_1_5
参数含义与`qwen_vl_utils<0.0.12`或者`qwen_omni_utils`库中含义一致，可以查看[这里](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24)。ms-swift通过修改这些常数值来控制图片分辨率和视频帧率，避免训练时OOM。

- IMAGE_FACTOR: 默认为28。
- MIN_PIXELS: 默认为`4 * 28 * 28`。图像的最小分辨率。建议设置为28*28的倍数。
- 🔥MAX_PIXELS: 默认为`16384 * 28 * 28`。图像的最大分辨率。建议设置为28*28的倍数。
- MAX_RATIO: 默认为200。
- VIDEO_MIN_PIXELS: 默认为`128 * 28 * 28`。视频中一帧的最小分辨率。建议设置为28*28的倍数。
- 🔥VIDEO_MAX_PIXELS: 默认为`768 * 28 * 28`。视频中一帧的最大分辨率。建议设置为28*28的倍数。
- VIDEO_TOTAL_PIXELS: 默认为`24576 * 28 * 28`。
- FRAME_FACTOR: 默认为2。
- FPS: 默认为2.0。
- FPS_MIN_FRAMES: 默认为4。一段视频的最小抽帧数。
- 🔥FPS_MAX_FRAMES: 默认为768。一段视频的最大抽帧数。
- 🔥QWENVL_BBOX_FORMAT: (ms-swift>=3.9.1) grounding格式使用'legacy'还是'new'。'legacy'格式为：`<|object_ref_start|>一只狗<|object_ref_end|><|box_start|>(432,991),(1111,2077)<|box_end|>`，'new'格式参考：[Qwen3-VL cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)，并参考[grounding数据集格式文档](../Customization/Custom-dataset.md#grounding)。默认为'legacy'。
  - 注意：该环境变量适配Qwen2/2.5/3-VL和Qwen2.5/3-Omni系列模型。

### qwen2_audio
- SAMPLING_RATE: 默认为16000。

### qwen2_5_omni, qwen3_omni
qwen2_5_omni除了包含qwen2_5_vl和qwen2_audio的模型特定参数外，还包含以下参数：
- USE_AUDIO_IN_VIDEO: 默认为False。是否使用video中的音频信息。
- 🔥ENABLE_AUDIO_OUTPUT: 默认为None，即使用`config.json`中的值。若使用zero3进行训练，请设置为False。
  - 提示：ms-swift只对thinker部分进行微调，建议设置为False以降低显存占用（只创建thinker部分的模型结构）。

### qwen3_vl
参数含义与`qwen_vl_utils>=0.0.14`库中的含义一致，可以查看[这里](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24)。通过传入以下环境变量，可以修改该库的全局变量默认值。（也兼容使用`qwen2_5_vl`的环境变量，例如：`MAX_PIXELS`、`VIDEO_MAX_PIXELS`，会做自动转换。）

- SPATIAL_MERGE_SIZE: 默认为2。
- IMAGE_MIN_TOKEN_NUM: 默认为`4`，代表一张图片最小图像tokens的个数。
- 🔥IMAGE_MAX_TOKEN_NUM: 默认为`16384`，代表一张图片最大图像tokens的个数。（用于避免OOM）
  - 提示：等价最大图像像素为`IMAGE_MAX_TOKEN_NUM * 32 *32`。
- VIDEO_MIN_TOKEN_NUM: 默认为`128`，代表视频中一帧的最小视频tokens的个数。
- 🔥VIDEO_MAX_TOKEN_NUM: 默认为`768`，代表视频中一帧的最大视频tokens的个数。（用于避免OOM）
- MAX_RATIO: 默认为200。
- FRAME_FACTOR: 默认为2。
- FPS: 默认为2.0。
- FPS_MIN_FRAMES: 默认为4。代表一段视频的最小抽帧数。
- 🔥FPS_MAX_FRAMES: 默认为768，代表一段视频的最大抽帧数。（用于避免OOM）


### qwen3_vl_emb, qwen3_vl_reranker
参数含义与`qwen3_vl`相同，见上面的描述。以下为对默认值的覆盖：

- IMAGE_MAX_TOKEN_NUM: qwen3_vl_emb默认为1800, qwen3_vl_reranker默认为1280。具体参考这里：[qwen3_vl_embedding](https://modelscope.cn/models/Qwen/Qwen3-VL-Embedding-2B/file/view/master/scripts%2Fqwen3_vl_embedding.py?status=1#L26), [qwen3_vl_reranker](https://modelscope.cn/models/Qwen/Qwen3-VL-Reranker-2B/file/view/master/scripts%2Fqwen3_vl_reranker.py?status=1#L16)。
- FPS: 默认为1。
- FPS_MAX_FRAMES: 默认为64。


### internvl, internvl_phi3
参数含义可以查看[这里](https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)。
- MAX_NUM: 默认为12。
- INPUT_SIZE: 默认为448。

### internvl2, internvl2_phi3, internvl2_5, internvl3, internvl3_5
参数含义可以查看[这里](https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B)。
- MAX_NUM: 默认为12。
- INPUT_SIZE: 默认为448。
- VIDEO_MAX_NUM: 默认为1。视频的MAX_NUM。
- VIDEO_SEGMENTS: 默认为8。


### minicpmv2_6, minicpmo2_6, minicpmv4
- MAX_SLICE_NUMS: 默认为9，参考[这里](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6/file/view/master?fileName=config.json&status=1)。
- VIDEO_MAX_SLICE_NUMS: 默认为1，视频的MAX_SLICE_NUMS，参考[这里](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)。
- MAX_NUM_FRAMES: 默认为64，参考[这里](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)。

### minicpmo2_6
- INIT_TTS: 默认为False。
- INIT_AUDIO: 默认为False。

### ovis1_6, ovis2
- MAX_PARTITION: 默认为9，参考[这里](https://github.com/AIDC-AI/Ovis/blob/d248e34d755a95d24315c40e2489750a869c5dbc/ovis/model/modeling_ovis.py#L312)。

### ovis2_5
以下参数含义可以在[这里](https://modelscope.cn/models/AIDC-AI/Ovis2.5-2B)的示例代码中找到。
- MIX_PIXELS: int类型，默认为`448 * 448`。
- MAX_PIXELS: int类型，默认为`1344 * 1792`。若出现OOM，可以调小该值。
- VIDEO_MAX_PIXELS: int类型，默认为`896 * 896`。
- NUM_FRAMES: 默认为8。用于视频抽帧。

### mplug_owl3, mplug_owl3_241101
- MAX_NUM_FRAMES: 默认为16，参考[这里](https://modelscope.cn/models/iic/mPLUG-Owl3-7B-240728)。

### xcomposer2_4khd
- HD_NUM: 默认为55，参考[这里](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b)。

### xcomposer2_5
- HD_NUM: 图片数量为1时，默认值为24。大于1，默认为6。参考[这里](https://modelscope.cn/models/AI-ModelScope/internlm-xcomposer2d5-7b/file/view/master?fileName=modeling_internlm_xcomposer2.py&status=1#L254)。

### video_cogvlm2
- NUM_FRAMES: 默认为24，参考[这里](https://github.com/zai-org/CogVLM2/blob/main/video_demo/inference.py#L22)。

### phi3_vision
- NUM_CROPS: 默认为4，参考[这里](https://modelscope.cn/models/LLM-Research/Phi-3.5-vision-instruct)。

### llama3_1_omni
- N_MELS: 默认为128，参考[这里](https://github.com/ictnlp/LLaMA-Omni/blob/544d0ff3de8817fdcbc5192941a11cf4a72cbf2b/omni_speech/infer/infer.py#L57)。

### video_llava
- NUM_FRAMES: 默认为16。


## 其他环境变量
- CUDA_VISIBLE_DEVICES: 控制使用哪些GPU卡。默认使用所有卡。
- ASCEND_RT_VISIBLE_DEVICES: 控制使用哪些NPU卡（只对ASCEND卡生效）。默认使用所有卡。
- MODELSCOPE_CACHE: 控制缓存路径。（多机训练时建议设置该值，以确保不同节点使用相同的数据集缓存）
- PYTORCH_CUDA_ALLOC_CONF: 推荐设置为`'expandable_segments:True'`，这将减少GPU内存碎片，具体请参考[torch文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)。
- NPROC_PER_NODE: torchrun中`--nproc_per_node`的参数透传。默认为1。若设置了`NPROC_PER_NODE`或者`NNODES`环境变量，则使用torchrun启动训练或推理。
- MASTER_PORT: torchrun中`--master_port`的参数透传。默认为29500。
- MASTER_ADDR: torchrun中`--master_addr`的参数透传。
- NNODES: torchrun中`--nnodes`的参数透传。
- NODE_RANK: torchrun中`--node_rank`的参数透传。
- LOG_LEVEL: 日志的level，默认为'INFO'，你可以设置为'WARNING', 'ERROR'等。
- SWIFT_DEBUG: 在`engine.infer(...)`时，若设置为'1'，TransformersEngine将会打印input_ids和generate_ids的内容方便进行调试与对齐。
- VLLM_USE_V1: 用于切换vLLM使用V0/V1版本。
- SWIFT_TIMEOUT: (ms-swift>=3.10) 若多模态数据集中存在图像URL，该参数用于控制获取图片的timeout，默认为20s。
- ROOT_IMAGE_DIR: (ms-swift>=3.8) 图像（多模态）资源的根目录。通过设置该参数，可以在数据集中使用相对于 `ROOT_IMAGE_DIR` 的相对路径。默认情况下，是相对于运行目录的相对路径。
- SWIFT_SINGLE_DEVICE_MODE: (ms-swift>=3.10) 单设备模式，可选值为"0"(默认值)/"1"，在此模式下，每个进程只能看到一个设备
- SWIFT_PATCH_CONV3D: (ms-swift>=3.11.2) 若使用torch==2.9，会遇到Conv3d运行缓慢的问题，可以通过设置`SWIFT_PATCH_CONV3D=1`规避该问题，具体查看[这个issue](https://github.com/modelscope/ms-swift/issues/7108)。
