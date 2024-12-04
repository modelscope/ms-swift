# ReleaseNote

## 新功能

1. 对数据集进行了整体重构，目前数据集加载整体速度提升2~20倍，encode速度提升2~4倍，且完整支持streaming模式
    - 使用--load_from_cache_file true来支持使用数据前处理缓存
    - 使用--dataset_num_proc来支持多进程加速处理
    - 使用--streaming来支持流式加载数据集
2. 对模型进行了重构，移除了2.x的model_type机制，目前可以直接使用--model xxx/xxx来训练和推理
    - 如果是新模型，可以直接使用--model xxx/xxx --template xxx --model_type xxx，而无需写python脚本进行模型注册了
3. template支持了jinja模式进行推理，使用--template_backend jinja来使用transformers官方模板
4. app-ui合并至web-ui，因此app-ui能力支持了多模态推理
5. 支持了plugin机制，用于定制训练过程，目前支持的plugin有：
    - callback 定制训练回调方法
    - custom_trainer 定制trainer
    - loss 定制loss方法
    - loss_scale 定制每个token的权重
    - metric 定制交叉验证的指标
    - optimizer 定制训练使用的optimizer和lr_scheduler
    - tools 定制agent训练的system格式
    - tuner 定制新的tuner
6. 支持All-to-All模型，即Emu3-Gen或Janus等文生图或全模态模型的训练和部署等
7. 对examples进行了功能提升，目前examples可以全面反映SWIFT的能力，易用性更强
8. 支持了一行命令启动多机训练，详情查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/deepspeed/README.md).
9. streaming模式可以高效率运行，并支持了--packing命令以获得更稳定的训练效率
10. 多模态人类对齐支持KTO算法
11. 可以使用--use_hf 1/0来切换hf社区和ms社区的数据集模型的下载上传
12. 更好地支持了以代码形式进行训练、推理，代码结构更清晰，并补充了大量的代码注释
13. 支持了pt backend下的批量推理和部署
14. 支持了基于deepspeed框架的多卡推理

## BreakChange

本文档列举3.x版本和2.x版本的BreakChange。开发者在使用时应当注意这些不同。

### 参数差异

- model_type的含义发生了变化。3.0版本需要指定--model或--ckpt_dir，model_type仅当模型为SWIFT不支持模型时才需要额外指定
- sft_type更名为train_type
- model_id_or_path更名为model
- template_type更名为template
- quantization_bit更名为quant_bits
- check_model_is_latest更名为check_model
- batch_size更名为per_device_train_batch_size，沿用了transformers的命名规则
- eval_batch_size更名为per_device_eval_batch_size，沿用了transformers的命名规则
- tuner_backend移除了swift选项
- use_flash_attn更名为attn_impl
- bnb_4bit_comp_dtype更名为bnb_4bit_compute_dtype
- 移除了train_dataset_sample和val_dataset_sample
- dtype更名为torch_dtype，同时选项名称从bf16变更为标准的bfloat16，fp16变更为float16，fp32变更为float32
- 移除了eval_human选项
- dataset选项移除了HF::使用方式，使用新增的--use_hf控制下载和上传
- 移除了do_sample选项，使用temperature进行控制
- add_output_dir_suffix更名为add_version
- 移除了eval_token，使用api_key支持
- target_modules(lora_target_modules)的ALL改为了all-linear，含义相同

2.0标记为compatible参数的部分整体移除了。

### 功能

1. 预训练请使用swift pt命令。该命令会默认使用generation template，而swift sft命令默认使用model_type预置的template
2. 整体移除了2.x版本的examples目录，并添加了按功能类型划分的新examples
3. 数据集格式完全向`messages`格式兼容，不再支持query/response/history格式的自定义数据集
4. merge_lora的存储目录可以通过`--output_dir`指定了，且merge_lora和量化不能在一个命令中执行，需要最少两个命令
5. 移除了app-ui界面，并使用`swift web-ui --model xxx`进行替代，并支持了多模态界面部署
6. 移除了AIGC的依赖以及对应的examples和训练代码

## 待完成

1. RM/PPO能力3.0版本尚不支持，请使用2.6.1版本
2. 自定义数据集评测3.0版本尚不支持，请使用2.6.1版本
3. Megatron预训练能力3.0版本尚不支持，请使用2.6.1版本
