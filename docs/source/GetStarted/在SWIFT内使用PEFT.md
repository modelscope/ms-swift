# 对Peft的兼容性

为了支持习惯Peft的用户，Swift提供了对于Peft的兼容性。用户可以从swift中import peft组件：

>PeftModel
>
>PeftConfig
>
>PeftModelForSeq2SeqLM
>
>PeftModelForSequenceClassification
>
>PeftModelForTokenClassification
>
>PeftModelForCausalLM
>
>PromptEncoderConfig
>
>PromptTuningConfig
>
>PrefixTuningConfig
>
>PromptLearningConfig
>
>LoraConfig
>
>get_peft_config
>
>get_peft_model_state_dict
>
>get_peft_model

以上组件均可以从swift中import：

```python
from swift import PeftModel, PeftConfig
```

Swift类也支持初始化Peft的tuner：

```python
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.models.nlp.structbert import SbertConfig

from swift import LoraConfig, Swift
model = SbertForSequenceClassification(SbertConfig())
lora_config = LoraConfig(target_modules=['query', 'key', 'value'])
model = Swift.prepare_model(model, lora_config)
```

Swift对Peft进行了浅封装，使Peft可以在from_pretrained时使用modelscope hub中的模型。
