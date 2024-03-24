# Compatibility with Peft

To support users accustomed to Peft, Swift provides compatibility with Peft. Users can import Peft components from Swift:

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

All of the above components can be imported from Swift:

```python
from swift import PeftModel, PeftConfig
```

The Swift class also supports initializing Peft's tuner:

```python
from modelscope.models.nlp import SbertForSequenceClassification
from modelscope.models.nlp.structbert import SbertConfig

from swift import LoraConfig, Swift
model = SbertForSequenceClassification(SbertConfig())
lora_config = LoraConfig(target_modules=['query', 'key', 'value'])
model = Swift.prepare_model(model, lora_config)
```

Swift provides a shallow wrapper for Peft, allowing Peft to use models from the modelscope hub when calling from_pretrained.
