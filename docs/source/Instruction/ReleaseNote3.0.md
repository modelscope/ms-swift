# ReleaseNote 3.0

> 如果您在3.x版本使用上遇到任何问题，请提交issue给我们。如存在2.x可用而3.x不可用的情况请暂时使用2.x版本等待我们修复完成。

## 新功能

1. 数据集模块重构。数据集加载速度提升2-20倍，encode速度提升2-4倍，支持streaming模式
    - 移除了dataset_name机制，采用dataset_id、dataset_dir、dataset_path方式指定数据集
    - 使用`--dataset_num_proc`支持多进程加速处理
    - 使用`--streaming`支持流式加载hub端和本地数据集
    - 支持`--packing`命令以获得更稳定的训练效率
    - 指定`--dataset <dataset_dir>`支持本地加载开源数据集
2. 对模型进行了重构：
    - 移除了model_type机制，使用`--model <model_id>/<model_path>`来训练和推理
    - 若是新模型，直接使用`--model <model_id>/<model_path> --template xxx --model_type xxx`，无需书写python脚本进行模型注册
3. template模块重构：
    - 使用`--template_backend jinja`采用jinja模式推理
    - 采用messages格式作为入参接口
4. 支持了plugin机制，用于定制训练过程，目前支持的plugin有：
    - callback 定制训练回调方法
    - loss 定制loss方法
    - loss_scale 定制每个token的权重
    - metric 定制交叉验证的指标
    - optimizer 定制训练使用的optimizer和lr_scheduler
    - tools 定制agent训练的system格式
    - tuner 定制新的tuner
4. 训练模块重构：
    - 支持了一行命令启动多机训练，详情查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/deepspeed/README.md)
    - 支持所有多模态LLM的PreTrain
    - 训练中的predict_with_generate采用infer模块，支持多模态LLM和多卡
    - 人类对齐KTO算法支持多模态LLM
5. 推理与部署模块重构：
    - 支持pt backend下的batch推理，支持多卡推理
    - 推理和部署模块统一采用openai格式接口
    - 支持了异步推理接口
6. app-ui合并入web-ui，app-ui支持多模态推理
7. 支持All-to-All模型，即Emu3-Gen或Janus等文生图或全模态模型的训练和部署等
8. 对examples进行了功能提升，目前examples可以全面反映SWIFT的能力，易用性更强
9. 使用`--use_hf true/false`来切换HuggingFace社区和ModelScope社区的数据集模型的下载上传
10. 更好地支持了以代码形式进行训练、推理，代码结构更清晰，并补充了大量的代码注释


## BreakChange

本文档列举3.x版本和2.x版本的BreakChange。开发者在使用时应当注意这些不同。

### 参数差异

- model_type的含义发生了变化。3.0版本只需要指定--model，model_type仅当模型为SWIFT不支持模型时才需要额外指定
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
- deepspeed的配置更改为`default-zero2`->`zero2`, `default-zero3`->`zero3`
- infer/deploy/export移除了--ckpt_dir参数，使用--model, --adapters进行控制

2.0标记为compatible参数的部分整体移除了。

### 功能

1. 预训练请使用swift pt命令。该命令会默认使用generation template，而swift sft命令默认使用model_type预置的template
2. 整体移除了2.x版本的examples目录，并添加了按功能类型划分的新examples
3. 数据集格式完全向messages格式兼容，不再支持query/response/history格式
4. merge_lora的存储目录可以通过`--output_dir`指定了，且merge_lora和量化不能在一个命令中执行，需要最少两个命令
5. 使用`swift app --model xxx`开启app-ui界面，支持了多模态界面推理
6. 移除了AIGC的依赖以及对应的examples和训练代码

## 待完成

1. 自定义数据集评测3.0版本尚不支持，请使用2.6.1版本
2. Megatron预训练能力3.0版本尚不支持，请使用2.6.1版本
3. 文档和README暂时未更新完整
