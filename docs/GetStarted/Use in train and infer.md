# Swift API

## 在训练中使用Swift

调用`Swift.prepare_model()`来将tuners添加到模型上：

```python
from modelscope import Model
from swift import Swift, LoRAConfig
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
lora_config = LoRAConfig(
                r=16,
                target_modules=['query_key_value'],
                lora_alpha=32,
                lora_dropout=0.)
model = Swift.prepare_model(model, lora_config)
# use model to do other things
```

也可以同时使用多个tuners：

```python
from modelscope import Model
from swift import Swift, LoRAConfig, AdapterConfig
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
lora_config = LoRAConfig(
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

## 在推理时使用Swift

使用`Swift.from_pretrained()`来拉起训练后存储的checkpoint：

```python
from modelscope import Model
from swift import Swift
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
model = Swift.from_pretrained(model, './output')
```

## 加载多个tuners并在不同线程中并行使用

在模型提供服务时，很可能出现一个模型同时服务多个http线程的情况，其中每个线程代表了一类用户请求。Swift支持在不同线程中激活不同tuners：

```python
from modelscope import Model
from swift import Swift
import torch
model = Model.from_pretrained('ZhipuAI/chatglm2-6b', torch_dtype=torch.bfloat16, device_map='auto')
# 假设output中存在训练完成的a、b、c、d是个tuners
model = Swift.from_pretrained(model, './output')

# 假设两类请求，一类使用a、b两个tuner，一类使用c、d两个tuner
type_1 = ['a', 'b', 'c']
type_2 = ['a', 'c', 'd']

def request(_input, _type):
  if _type == 'type_1':
    model.set_active_adapters(type_1)
  elif _type == 'type_2':
    model.set_active_adapters(type_2)
  return model(**_input)

```

在不同线程中使用同一个tuner是安全的。
