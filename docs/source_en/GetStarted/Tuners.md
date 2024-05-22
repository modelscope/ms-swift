# Basic Usage

"Tuner" refers to additional structures attached to a model to reduce the number of training parameters or improve training accuracy. Currently, SWIFT supports the following tuners:

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
13. Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](docs/source/GetStarted/ResTuning.md) >
14. Tuners provided by [PEFT](https://github.com/huggingface/peft), such as IA3, AdaLoRA, etc.

## Using in Training

Call `Swift.prepare_model()` to add tuners to the model:

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

Multiple tuners can also be used simultaneously:

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

When using multiple tuners, the second parameter should be a Dict where the key is the tuner name and the value is the tuner configuration.

After training, you can call:

```python
model.save_pretrained(save_directory='./output')
```

to store the model checkpoint. The model checkpoint file will only include the weights of the tuners, not the weights of the model itself. The stored structure is as follows:

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

If only a single config is passed in, the default name `default` will be used:

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

### Complete Training Code

```python
# A100 18G memory
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import MsDataset, AutoTokenizer
from modelscope import AutoModelForCausalLM
from swift import Swift, LoraConfig
from swift.llm import get_template, TemplateType
import torch

# load model
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

## Using in Inference

Use `Swift.from_pretrained()` to load the stored checkpoint:

```python
from modelscope import Model
from swift import Swift
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
model = Swift.from_pretrained(model, './output')
```

### Complete Inference Code

```python
# A100 14G memory
import torch
from modelscope import AutoModelForCausalLM, GenerationConfig
from modelscope import AutoTokenizer

from swift import Swift
from swift.llm import get_template, TemplateType, to_device

# load model
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

# Interface List

## Swift Class Static Interfaces

- `Swift.prepare_model(model, config, **kwargs)`
  - Explain: Load a tuner onto the model. If it is a subclass of PeftConfig, use the corresponding interface of the Peft library to load the tuner. When using SwiftConfig, this interface can accept a SwiftModel instance and be called repeatedly, which has the same effect as passing a dictionary to config.
    - This interface supports parallel loading of multiple tuners of different types for simultaneous use.
  - Parameters:
    - `model`: An instance of `torch.nn.Module` or `SwiftModel`, the model to be loaded
    - `config`: An instance of `SwiftConfig`, `PeftConfig`, or a dictionary of custom tuner names to configs
  - Return value: An instance of `SwiftModel` or `PeftModel`
- `Swift.merge_and_unload(model)`
  - Explain: Merge the LoRA weights back into the original model and completely unload the LoRA part
  - Parameters:
    - model: An instance of `SwiftModel` or `PeftModel`, the model instance with LoRA loaded
  - Return value: None

- `Swift.merge(model)`

  - Explain: Merge the LoRA weights back into the original model without unloading the LoRA part

  - Parameters:
    - model: An instance of `SwiftModel` or `PeftModel`, the model instance with LoRA loaded

  - Return value: None

- `Swift.unmerge(model)`

  - Explain: Split the LoRA weights from the original model weights back into the LoRA structure

  - Parameters:
    - model: An instance of `SwiftModel` or `PeftModel`, the model instance with LoRA loaded

  - Return value: None

- `Swift.save_to_peft_format(ckpt_dir, output_dir)`

  - Explain: Convert the stored LoRA checkpoint to a Peft compatible format. The main changes are:

    - `default` will be split from the corresponding `default` folder into the output_dir root directory
    - The `{tuner_name}.` field in weights will be removed, for example `model.layer.0.self.in_proj.lora_A.default.weight` will become `model.layer.0.self.in_proj.lora_A.weight`
    - The prefix `basemodel.model` will be added to the keys in weights

    - Note: Only LoRA can be converted, other types of tuners cannot be converted due to Peft itself not supporting them. Additionally, when there are extra parameters like `dtype` set in LoRAConfig, it does not support conversion to Peft format. In this case, you can manually delete the corresponding fields in adapter_config.json

  - Parameters:

    - ckpt_dir: Original weights directory
    - output_dir: Target weights directory

  - Return value: None

- `Swift.from_pretrained(model, model_id, adapter_name, revision, **kwargs)`
  - Explain: Load tuners from the stored weights directory onto the model. If adapter_name is not passed, all tuners under the model_id directory will be loaded. Same as `prepare_model`, this interface can be called repeatedly.
  - Parameters:
    - model: An instance of `torch.nn.Module` or `SwiftModel`, the model to be loaded
    - model_id: `str` type, the tuner checkpoint to be loaded, can be a ModelScope hub id or a local directory produced by training
    - adapter_name: `str` or `List[str]` or `Dict[str, str]` type or `None`, the tuner name in the tuner directory to be loaded. If `None`, all named tuners will be loaded. If `str` or `List[str]`, only certain specific tuners will be loaded. If `Dict`, the tuner indicated by `key` will be loaded and renamed to `value`.
    - revision: If model_id is a ModelScope id, revision can specify the corresponding version number

## SwiftModel Interface

The following lists the interfaces that users may call. Other internal interfaces or interfaces not recommended for use can be viewed through the `make docs` command to generate the API Doc documentation.

- `SwiftModel.create_optimizer_param_groups(self, **defaults)`
  - Explain: Create parameter groups based on the loaded tuners, currently only effective for the `LoRA+` algorithm
  - Parameters:
    - defaults: Default parameters for `optimizer_groups`, such as `lr` and `weight_decay`
  - Return value:
    - The created `optimizer_groups`

- `SwiftModel.add_weighted_adapter(self, ...)`
  - Explain: Merge existing LoRA tuners into one
  - Parameters:
    - This interface is a transparent pass-through of PeftModel.add_weighted_adapter, parameters can refer to: [add_weighted_adapter documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter)

- `SwiftModel.save_pretrained(self, save_directory, safe_serialization, adapter_name)`
  - Explain: Store tuner weights
  - Parameters:
    - save_directory: Storage directory
    - safe_serialization: Whether to use safe_tensors, default is False
    - adapter_name: The adapter tuner to store, if not passed, all tuners will be stored by default
- `SwiftModel.set_active_adapters(self, adapter_names, offload=None)`
  - Explain: Set the currently active adapters, adapters not in the list will be deactivated
    - In `inference`, the environment variable `USE_UNIQUE_THREAD=0/1` is supported, default value is `1`. If `0`, set_active_adapters only takes effect for the current thread. In this case, the tuners activated by this thread are used by default, and tuners in different threads do not interfere with each other.
  - Parameters:
    - adapter_names: Activated tuners
    - offload: How to handle deactivated adapters, default is `None` which means leave them in GPU memory. Both `cpu` and `meta` are supported, indicating offloading to cpu and meta devices to reduce GPU memory consumption. When `USE_UNIQUE_THREAD=0`, do not pass a value to offload to avoid affecting other threads.
  - Return value: None
- `SwiftModel.activate_adapter(self, adapter_name)`
  - Explain: Activate a tuner
    - In `inference`, the environment variable `USE_UNIQUE_THREAD=0/1` is supported, default value is `1`. If `0`, activate_adapter only takes effect for the current thread. In this case, the tuners activated by this thread are used by default, and tuners in different threads do not interfere with each other.
  - Parameters:
    - adapter_name: The name of the tuner to activate
  - Return value: None
- `SwiftModel.deactivate_adapter(self, adapter_name, offload)`
  - Explain: Deactivate a tuner
    - When the environment variable `USE_UNIQUE_THREAD=0`, do not call this interface
  - Parameters:
    - adapter_name: The name of the tuner to deactivate
    - offload: How to handle deactivated adapters, default is `None` which means leave them in GPU memory. Both `cpu` and `meta` are supported, indicating offloading to cpu and meta devices to reduce GPU memory consumption
  - Return value: None

- `SwiftModel.get_trainable_parameters(self)`

  - Explain: Return training parameter information

  - Parameters: None

  - Return value: Training parameter information, format is as follows:
    ```text
    trainable params: 100M || all params: 1000M || trainable%: 10.00% || cuda memory: 10GiB.
    ```
