# 使用Tuners

tuner是指附加在模型上的额外结构部分，用于减少训练参数量或者提高训练精度。目前SWIFT支持的tuners有：

- LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
- LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf)
- LLaMA PRO: [LLAMA PRO: Progressive LLaMA with Block Expansion](https://arxiv.org/pdf/2401.02415.pdf)
- GaLore/Q-GaLore: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- Liger Kernel: [Liger Kernel: Efficient Triton Kernels for LLM Training](https://arxiv.org/abs/2410.10989)
- LISA: [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919)
- UnSloth: https://github.com/unslothai/unsloth
- SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  |  [Project Page](https://scedit.github.io/) >
- NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
- LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
- Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
- Vision Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
- Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
- Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](ResTuning.md) >
- [PEFT](https://github.com/huggingface/peft)提供的tuners, 如AdaLoRA、DoRA、Fourierft等

## 接口列表

### Swift类静态接口

- `Swift.prepare_model(model, config, **kwargs)`
  - 接口作用：加载某个tuner到模型上，如果是PeftConfig的子类，则使用Peft库的对应接口加载tuner。在使用SwiftConfig的情况下，本接口可以传入SwiftModel实例并重复调用，此时和config传入字典的效果相同。
    - 本接口支持并行加载不同类型的多个tuners共同使用
  - 参数：
    - `model`: `torch.nn.Module`或`SwiftModel`的实例，被加载的模型
    - `config`: `SwiftConfig`、`PeftConfig`的实例，或者一个自定义tuner名称对config的字典
  - 返回值：`SwiftModel`或`PeftModel`的实例
- `Swift.merge_and_unload(model)`
  - 接口作用：将LoRA weights合并回原模型，并将LoRA部分完全卸载
  - 参数：
    - model: `SwiftModel`或`PeftModel`的实例，已加载LoRA的模型实例
  - 返回值：None

- `Swift.merge(model)`

  - 接口作用：将LoRA weights合并回原模型，不卸载LoRA部分

  - 参数：
    - model: `SwiftModel`或`PeftModel`的实例，已加载LoRA的模型实例

  - 返回值：None

- `Swift.unmerge(model)`

  - 接口作用：将LoRA weights从原模型weights中拆分回LoRA结构

  - 参数：
    - model: `SwiftModel`或`PeftModel`的实例，已加载LoRA的模型实例

  - 返回值：None

- `Swift.save_to_peft_format(ckpt_dir, output_dir)`

  - 接口作用：将存储的LoRA checkpoint转换为Peft兼容的格式。主要改变有：

    - `default`会从对应的`default`文件夹中拆分到output_dir根目录中
    - weights中的`{tuner_name}.`字段会被移除，如`model.layer.0.self.in_proj.lora_A.default.weight`会变为`model.layer.0.self.in_proj.lora_A.weight`
    - weights中的key会增加`basemodel.model`前缀

    - 注意：只有LoRA可以被转换，其他类型tuner由于Peft本身不支持，因此会报转换错误。此外，由于LoRAConfig中存在额外参数，如`dtype`，因此在这些参数有设定的情况下，不支持转换为Peft格式，此时可以手动删除adapter_config.json中的对应字段

  - 参数：

    - ckpt_dir：原weights目录
    - output_dir：目标weights目录

  - 返回值：None

- `Swift.from_pretrained(model, model_id, adapter_name, revision, **kwargs)`
  - 接口作用：从存储的weights目录中加载起tuner到模型上，如果adapter_name不传，则会将model_id目录下所有的tuners都加载起来。同`prepare_model`相同，本接口可以重复调用
  - 参数：
    - model：`torch.nn.Module`或`SwiftModel`的实例，被加载的模型
    - model_id：`str`类型，待加载的tuner checkpoint， 可以是魔搭hub的id，或者训练产出的本地目录
    - adapter_name：`str`或`List[str]`或`Dict[str, str]`类型或`None`，待加载tuner目录中的tuner名称，如果为`None`则加载所有名称的tuners，如果是`str`或`List[str]`则只加载某些具体的tuner，如果是`Dict`，则将`key`指代的tuner加载起来后换成`value`的名字
    - revision: 如果model_id是魔搭的id，则revision可以指定对应版本号

### SwiftModel接口

下面列出用户可能调用的接口列表，其他内部接口或不推荐使用的接口可以通过`make docs`命令查看API Doc文档。

- `SwiftModel.create_optimizer_param_groups(self, **defaults)`
  - 接口作用：根据加载的tuners创建parameter groups，目前仅对`LoRA+`算法有作用
  - 参数：
    - defaults：`optimizer_groups`的默认参数，如`lr`和`weight_decay`
  - 返回值：
    - 创建的`optimizer_groups`

- `SwiftModel.add_weighted_adapter(self, ...)`
  - 接口作用：将已有的LoRA tuners合并为一个
  - 参数：
    - 本接口是PeftModel.add_weighted_adapter的透传，参数可以参考：[add_weighted_adapter文档](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter)

- `SwiftModel.save_pretrained(self, save_directory, safe_serialization, adapter_name)`
  - 接口作用：存储tuner weights
  - 参数：
    - save_directory：存储目录
    - safe_serialization： 是否使用safe_tensors，默认为False
    - adapter_name：存储的adapter tuner，如果不传则默认存储所有的tuners
- `SwiftModel.set_active_adapters(self, adapter_names, offload=None)`
  - 接口作用：设置当前激活的adapters，不在列表中的adapters会被失活
    - 在`推理`时支持环境变量`USE_UNIQUE_THREAD=0/1`，默认值`1`，如果为`0`则set_active_adapters只对当前线程生效，此时默认使用本线程激活的tuners，不同线程tuners互不干扰
  - 参数：
    - adapter_names：激活的tuners
    - offload：失活的adapters如何处理，默认为`None`代表留在显存中，同时支持`cpu`和`meta`，代表offload到cpu和meta设备中以减轻显存消耗，在`USE_UNIQUE_THREAD=0`时offload不要传值以免影响其他线程
  - 返回值：None
- `SwiftModel.activate_adapter(self, adapter_name)`
  - 接口作用：激活一个tuner
    - 在`推理`时支持环境变量`USE_UNIQUE_THREAD=0/1`，默认值`1`，如果为`0`则activate_adapter只对当前线程生效，此时默认使用本线程激活的tuners，不同线程tuners互不干扰
  - 参数：
    - adapter_name：待激活的tuner名字
  - 返回值：None
- `SwiftModel.deactivate_adapter(self, adapter_name, offload)`
  - 接口作用：失活一个tuner
    - 在`推理`时环境变量`USE_UNIQUE_THREAD=0`时不要调用本接口
  - 参数：
    - adapter_name：待失活的tuner名字
    - offload：失活的adapters如何处理，默认为`None`代表留在显存中，同时支持`cpu`和`meta`，代表offload到cpu和meta设备中以减轻显存消耗
  - 返回值：None

- `SwiftModel.get_trainable_parameters(self)`

  - 接口作用：返回训练参数信息

  - 参数：无

  - 返回值：训练参数信息，格式如下：
    ```text
    trainable params: 100M || all params: 1000M || trainable%: 10.00% || cuda memory: 10GiB.
    ```
