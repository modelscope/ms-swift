# 基本使用

tuner是指附加在模型上的额外结构部分，用于减少训练参数量或者提高训练精度。目前SWIFT支持的tuners有：

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf)
3. LLaMA PRO: [LLAMA PRO: Progressive LLaMA with Block Expansion](https://arxiv.org/pdf/2401.02415.pdf)
4. GaLore: [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
5. LISA: [LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning](https://arxiv.org/abs/2403.17919)
6. UnSloth: https://github.com/unslothai/unsloth
7. SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  |  [Project Page](https://scedit.github.io/) >
8. NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
9. LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
10. Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
11. Vision Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
12. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
13. Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](ResTuning.md) >
14. [PEFT](https://github.com/huggingface/peft)提供的tuners, 如IA3, AdaLoRA等

## 在训练中使用

调用`Swift.prepare_model()`来将tuners添加到模型上：

```python
from modelscope import Model
from swift import Swift, LoraConfig
import torch
model = Model.from_pretrained('ZhipuAI/chatglm3-6b', torch_dtype=torch.bfloat16, device_map='auto')
lora_config = LoraConfig(
                r=16,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.)
model = Swift.prepare_model(model, lora_config)
```

也可以同时使用多个tuners：

```python
from modelscope import Model
from swift import Swift, LoraConfig, AdapterConfig
import torch
model = Model.from_pretrained('ZhipuAI/chatglm3-6b', torch_dtype=torch.bfloat16, device_map='auto')
lora_config = LoraConfig(
                r=16,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.)
adapter_config = AdapterConfig(
                dim=model.config.hidden_size,
                target_modules=['mlp'],
                method_name='forward',
                hidden_pos=0,
                adapter_length=32,
            )
model = Swift.prepare_model(model, {'first_tuner': lora_config, 'second_tuner': adapter_config})
# use model to do other things
```

在使用多个tuners时，传入的第二个参数需要是Dict，key是tuner名字，value是tuner配置。

训练后可以调用：

```python
model.save_pretrained(save_directory='./output')
```

来存储模型checkpoint。模型的checkpoint文件只会包括tuners的权重，不会包含模型本身的权重。存储后的结构如下：

> outputs
>
> ​     |-- configuration.json
>
> ​     |-- first_tuner
>
> ​               |-- adapter_config.json
>
> ​               |-- adapter_model.bin
>
> ​     |-- second_tuner
>
> ​               |-- adapter_config.json
>
> ​               |-- adapter_model.bin
>
> ​     |-- ...

如果只传入单独的config，则会使用默认的名称`default`：

> outputs
>
> ​      |-- configuration.json
>
> ​      |-- default
>
> ​                |-- adapter_config.json
>
> ​                |-- adapter_model.bin
>
> ​      |-- ...

### 完整的训练代码

```python
# A100 18G memory
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType
import torch

# 拉起模型
model = AutoModelForCausalLM.from_pretrained('ZhipuAI/chatglm3-6b', torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
lora_config = LoraConfig(
                r=16,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.05)
model = Swift.prepare_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained('ZhipuAI/chatglm3-6b', trust_remote_code=True)
dataset = MsDataset.load('AI-ModelScope/alpaca-gpt4-data-en', split='train')
template = get_template(TemplateType.chatglm3, tokenizer, max_length=1024)

def encode(example):
    inst, inp, output = example['instruction'], example.get('input', None), example['output']
    if output is None:
        return {}
    if inp is None or len(inp) == 0:
        q = inst
    else:
        q = f'{inst}\n{inp}'
    example, kwargs = template.encode({'query': q, 'response': output})
    return example

dataset = dataset.map(encode).filter(lambda e: e.get('input_ids'))
dataset = dataset.train_test_split(test_size=0.001)

train_dataset, val_dataset = dataset['train'], dataset['test']


train_args = Seq2SeqTrainingArguments(
    output_dir='output',
    learning_rate=1e-4,
    num_train_epochs=2,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy='steps',
    save_strategy='steps',
    dataloader_num_workers=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer)

trainer.train()
```

## 在推理时使用

使用`Swift.from_pretrained()`来拉起训练后存储的checkpoint：

```python
from modelscope import Model
from swift import Swift
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
model = Swift.from_pretrained(model, './output')
```

### 完整的推理代码

```python
# A100 14G memory
import torch
from modelscope import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer

from swift import Swift
from swift.llm import get_template, TemplateType, to_device

# 拉起模型
model = AutoModelForCausalLM.from_pretrained('ZhipuAI/chatglm3-6b', torch_dtype=torch.bfloat16,
                                             device_map='auto', trust_remote_code=True)
model = Swift.from_pretrained(model, 'output/checkpoint-xxx')
tokenizer = AutoTokenizer.from_pretrained('ZhipuAI/chatglm3-6b', trust_remote_code=True)
template = get_template(TemplateType.chatglm3, tokenizer, max_length=1024)

examples, tokenizer_kwargs = template.encode({'query': 'How are you?'})
if 'input_ids' in examples:
    input_ids = torch.tensor(examples['input_ids'])[None]
    examples['input_ids'] = input_ids
    token_len = input_ids.shape[1]

generation_config = GenerationConfig(
    max_new_tokens=1024,
    temperature=0.3,
    top_k=25,
    top_p=0.8,
    do_sample=True,
    repetition_penalty=1.0,
    num_beams=10,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id)

device = next(model.parameters()).device
examples = to_device(examples, device)

generate_ids = model.generate(
    generation_config=generation_config,
    **examples)
generate_ids = template.get_generate_ids(generate_ids, token_len)
print(tokenizer.decode(generate_ids, **tokenizer_kwargs))
# I'm an AI language model, so I don't have feelings or physical sensations. However, I'm here to assist you with any questions or tasks you may have. How can I help you today?
```

# 接口列表

## Swift类静态接口

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

## SwiftModel接口

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
