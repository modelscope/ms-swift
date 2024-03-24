# Customization and Extension
## Table of Contents
- [Custom Datasets](#custom-datasets) 
- [Custom Models](#custom-models)
- [Custom Dialogue Templates](#custom-dialogue-templates)

## Custom Datasets
We support two methods for **custom datasets**:

1. [Recommended]  **Command line arguments**: More convenient to support local custom datasets.
2. By **registering datasets**: More flexible, can further extend and develop swift, but requires some programming skills. Method 1 internally leverages method 2.

### 📌 [Recommended] Command line arguments 
You need to additionally specify in the sft.sh script:

```bash
--custom_train_dataset_path xxx.jsonl \ 
--custom_val_dataset_path yyy.jsonl \
```

The corresponding example sh script can be found [here](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/tongyi_finance_14b_chat_int4/qlora/sft.sh).

1. `--custom_train_dataset_path`: The default value is `[]`, indicating not to use a custom dataset. You can specify it in the following form: `--custom_train_dataset_path alpaca.csv` or specify multiple training datasets `--custom_train_dataset_path alpaca.csv chatml.jsonl swift.jsonl`, the script will automatically preprocess and concatenate them.

   > Training can be done by combining public datasets and custom datasets: `--dataset blossom-math-zh --custom_train_dataset_path custom_math.jsonl`.

2. `--custom_val_dataset_path`: The default value is `[]`, indicating not to use a custom validation dataset. If you specify `custom_train_dataset_path`, then the validation set of the custom dataset will be split according to the command line argument `dataset_test_ratio`.

The script supports file formats including `csv`, `json`, and `jsonl`. You need to ensure the passed in files conform to the following dataset formats. csv files only support instruction tuning, i.e. the case without history. json and jsonl files support system and history.

**Format 1:**

Pre-Training

```csv
response
11111 
aaaaa
AAAAA
```

```jsonl
{"response": "11111"}
{"response": "aaaaa"} 
{"response": "AAAAA"}
```

Single-Round Dialogue

```csv
query,response
11111,22222
aaaaa,bbbbb
AAAAA,BBBBB  
```

```jsonl
{"query": "11111", "response": "22222"}
{"query": "aaaaa", "response": "bbbbb"}
{"query": "AAAAA", "response": "BBBBB"}
```

Multi-Round Dialogue

```jsonl
{"query": "55555", "response": "66666"}
{"query": "eeeee", "response": "fffff", "history": []}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
```

```json
[{"query": "55555", "response": "66666"},
{"query": "eeeee", "response": "fffff", "history": []},
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}]
```

**Format 2:**

```jsonl
{"conversations": [{"from": "user", "value": "11111"}, {"from": "assistant", "value": "22222"}]} 
{"conversations": [{"from": "user", "value": "aaaaa"}, {"from": "assistant", "value": "bbbbb"}, {"from": "user", "value": "ccccc"}, {"from": "assistant", "value": "ddddd"}]}
{"conversations": [{"from": "user", "value": "AAAAA"}, {"from": "assistant", "value": "BBBBB"}, {"from": "user", "value": "CCCCC"}, {"from": "assistant", "value": "DDDDD"}]} 
```

**Format 3:**

```jsonl
{"messages": [{"role": "user", "content": "11111"}, {"role": "assistant", "content": "22222"}]}
{"messages": [{"role": "user", "content": "aaaaa"}, {"role": "assistant", "content": "bbbbb"}, {"role": "user", "content": "ccccc"}, {"role": "assistant", "content": "ddddd"}]} 
{"messages": [{"role": "user", "content": "AAAAA"}, {"role": "assistant", "content": "BBBBB"}, {"role": "user", "content": "CCCCC"}, {"role": "assistant", "content": "DDDDD"}]}
```

**Format 4:**

```csv
instruction,input,output
11111,22222,33333
aaaaa,bbbbb,ccccc
AAAAA,BBBBB,CCCCC
```

**Reinforcement Learning (DPO)**

```jsonl
{"query": "11111", "response": "22222", "rejected_response": "33333"}
{"query": "aaaaa", "response": "bbbbb", "rejected_response": "ccccc"} 
{"query": "AAAAA", "response": "BBBBB", "rejected_response": "CCCCC"}
```

### Registering Datasets

The following is an example of **registering datasets**. The complete py file can be viewed at [custom.py](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/custom.py), and the sh script can be viewed at [custom](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/custom).

```python
from typing import Optional, Tuple

from datasets import Dataset as HfDataset  
from modelscope import MsDataset

from swift.llm import get_dataset, register_dataset
from swift.utils import get_logger

logger = get_logger()


class CustomDatasetName:
    stsb_en = 'stsb-en'

def _preprocess_stsb(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}  
Similarity score: """
    query = []
    response = []
    for d in dataset:
        query.append(prompt.format(text1=d['text1'], text2=d['text2']))
        response.append(f"{d['label']:.1f}")
    return HfDataset.from_dict({'query': query, 'response': response})


@register_dataset(
    CustomDatasetName.stsb_en, 'huangjintao/stsb', task='text-generation')  
def get_stsb_dataset(dataset_id_or_path: str,
                     **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_dict = MsDataset.load(dataset_id_or_path)
    train_dataset = dataset_dict['train'].to_hf_dataset()
    val_dataset = dataset_dict['validation'].to_hf_dataset()
    return tuple(
        _preprocess_stsb(dataset) for dataset in [train_dataset, val_dataset])


if __name__ == '__main__':
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en], 
                                             check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')  
    print(f'val_dataset: {val_dataset}')

```

`register_dataset` will register the dataset in `DATASET_MAPPING`. The meaning of the parameters of this function are as follows:

- `dataset_name`: Required field, represents the name of the dataset, and is also the unique id of the dataset.

- `dataset_id_or_path`: Required field. Represents the `dataset_id` of the dataset on ModelScope Hub or the local `dataset_dir`.

- `get_function`: Default value is `None`. The function to get the dataset. If passed `None`, the decorator approach will be used to register the dataset. If passed a function, the normal approach will be used to register.
   > `get_function` needs to return `HfDataset` or `Tuple[HfDataset, Optional[HfDataset]]`. If only one dataset is returned, then that dataset is the train\_dataset, and the dataset processing function will split a portion of the dataset as val\_dataset (according to the command line argument `dataset_test_ratio`); if two datasets are returned, they will be used as train\_dataset and val\_dataset respectively. We support using multiple datasets for fine-tuning `get_dataset(['dataset1', 'dataset2'])`. We will concatenate the training and validation set portions of each sub-dataset respectively, and finally return the merged training and validation sets.

   > The `HfDataset` returned by the function needs to follow certain specifications. If you want to do **pre-training**, you only need to include the `response` field, please refer to the `'tigerbot-law-zh'` dataset for details. For **instruction tuning (single-round dialogue)**, the `query` and `response` fields need to be included, representing the user's query and the AI assistant's answer in instruction tuning respectively, please refer to the `'alpaca-zh'` dataset for details. For **multi-round dialogue**, an additional `history` field needs to be added, representing the historical information of the dialogue, please refer to the `'damo-agent-mini-zh'` dataset for details. If each dataset sample has a different `system`, an additional system field needs to be added, you can also refer to the `'damo-agent-mini-zh'` dataset for details.

- `**kwargs`: Other parameters used to annotate the dataset. This parameter generally does not need to be set.

## Custom Models
The following is an example of **custom models**. The complete py file can be viewed at [custom.py](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/custom.py), and the sh script can be viewed at [custom](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/custom).

```python
from typing import Any, Dict

from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torch import dtype as Dtype  
from transformers.utils.versions import require_version

from swift.llm import LoRATM, TemplateType, get_model_tokenizer, register_model
from swift.utils import get_logger

logger = get_logger()


class CustomModelType:
    tigerbot_7b = 'tigerbot-7b'
    tigerbot_13b = 'tigerbot-13b'
    tigerbot_13b_chat = 'tigerbot-13b-chat'


class CustomTemplateType:
    tigerbot = 'tigerbot'


@register_model(CustomModelType.tigerbot_7b,
                'TigerResearch/tigerbot-7b-base-v3', LoRATM.llama2,
                TemplateType.default_generation) 
@register_model(CustomModelType.tigerbot_13b,
                'TigerResearch/tigerbot-13b-base-v2', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b_chat, 
                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
                CustomTemplateType.tigerbot)
def get_tigerbot_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True') 
        model_kwargs['use_flash_attention_2'] = True
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer


if __name__ == '__main__':
    # test model base
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_7b, use_flash_attn=False)
    print(model.__class__.__name__)  
    # test model chat
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b_chat, use_flash_attn=False)
    print(model.__class__.__name__)
```

`register_model` will register the model in `MODEL_MAPPING`. The meaning of the parameters of this function are as follows:

- `model_type`: Required field. Represents the name of the model, and is also the unique id.
- `model_id_or_path`: Required field. Represents the `model_id` of the model in ModelScope Hub, or the local model directory `model_dir`.
- `lora_target_modules`: Default is `None`. Represents the default lora_target_modules to use when `--lora_target_modules DEFAULT` or `--lora_target_modules AUTO` is specified in the sh script, or when `--lora_target_modules` is not specified.
- `template`: Default is `TemplateType.default`. Represents the default dialogue template to use when `--template_type AUTO` is specified in the sh script, or when `--template_type` is not specified.
- `get_function`: Default value is `None`. The function to get model and tokenizer. If passed `None`, the decorator approach will be used to register the model. If passed a function, the normal approach will be used to register. 
- `requires`: Default is `[]`. Represents the dependencies required by the model that differ from other models. This parameter generally does not need to be set.
- `torch_dtype`: Default is `None`. Represents the recommended torch_dtype for the model to use. This parameter generally does not need to be set.
- `use_hf`: Default is `False`, i.e. set to modelscope hub. If you want to use huggingface hub, you can set it to True.
- `revision`: Default is `None`. Used to specify the version number of the model. If `use_hf` is False, it is set to 'master', if `use_hf` is True, it is set to 'main'. If `model_id_or_path` is a local model directory, this parameter is not effective. This parameter generally does not need to be set.
- `ignore_file_pattern`: Default is `None`. Represents the regular pattern of file names to be ignored when downloading, this parameter will be passed to `snapshot_download`. For example, `r'.+\.bin$'`, `r'.+\.savetensors$'`, etc. This parameter generally does not need to be set.
- `**kwargs`: Other parameters used to annotate model capabilities. This parameter generally does not need to be set.

## Custom Dialogue Templates 
The following is an example of **custom models**. The complete py file can be viewed at [custom.py](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/custom.py), and the sh script can be viewed at [custom](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/custom).

```python
from swift.llm import (Template, ModelType, dataset_map,  
                       get_model_tokenizer, get_template, get_dataset,
                       print_example, register_template, DatasetName)  
from swift.utils import get_logger

logger = get_logger()


class CustomTemplateType:
    tigerbot = 'tigerbot'


# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template(['{{SYSTEM}}'], ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [], 
             [['eos_token_id']]))

if __name__ == '__main__':
    # test template
    train_dataset, _ = get_dataset(DatasetName.blossom_math_zh)
    _, tokenizer = get_model_tokenizer(ModelType.qwen_7b_chat, load_model=False)
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)
```

`register_template` will register the dialogue template in `TEMPLATE_MAPPING`. The meaning of the parameters of this function are as follows:

- `template_type`: Required field, represents the name of the dialogue template, and is also the unique id of the template.
- `template`: Required field, needs to pass in a `Template`. To initialize `Template`, the following parameters need to be passed in: `prefix`, `prompt`, `chat_sep`, `suffix`, `default_system`.

The template initialization function will obtain the complete chat template based on these four contents. The meaning of these four configuration contents are as follows.

- `prefix`: Represents the prefix part of the dialogue template, generally including system part, prefix tokens, bos tokens, etc. We use `{{SYSTEM}}` as the placeholder for the system. If `{{SYSTEM}}` does not exist in the prefix, then this Template does not support system, e.g. `damo-agent-mini-zh` dataset.
- `prompt`: Represents a round of dialogue in the dialogue template. We use `{{QUERY}}` as the placeholder for the human query part in each round of dialogue, `{{ROUND0}}` represents the placeholder for which round of dialogue this is, starting from 0, and `{{ROUND1}}` starts from 1. The AI assistant's reply part will be concatenated after `prompt`, so we have not designed a placeholder for it. We will only calculate the loss for the AI assistant's reply part.
- `chat_sep`: If multi-round dialogue is needed, `chat_sep` will be used as the separator between each round of dialogue, such as: newline, etc. If set to None, then this Template does not support multi-round dialogue.
- `suffix`: Used as the suffix part of the dialogue template, generally eos token. Will be concatenated after the last round of dialogue.
- `default_system`: The default system.