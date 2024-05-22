# Customization and Extension
## Table of Contents
- [Custom Datasets](#custom-datasets)
- [Custom Models](#custom-models)
- [Custom Dialogue Templates](#custom-dialogue-templates)

## Custom Dataset

We support three methods for **customizing datasets**.

1. \[Recommended\] using command line arguments: It is more convenient to support custom datasets, and it supports four dataset formats (using `SmartPreprocessor`) as well as the `dataset_id` and `dataset_path`.
2. Adding datasets to `dataset_info.json` is more flexible than the first method, and supports using two preprocessors and specifying their parameters: `RenameColumnsPreprocessor`, `ConversationsPreprocessor` (default is to use `SmartPreprocessor`). You can directly modify the built-in `dataset_info.json` in Swift, or pass in an external json file using `--dataset_info_path xxx.json` (for users who prefer pip install over git clone to expand datasets).
3. Registering datasets: More flexible than the first two methods, it supports using functions to preprocess datasets. Methods 1 and 2 are implemented by leveraging method 3. You can directly modify the source code for expansion, or pass in a custom registration path using `--custom_register_path xxx.py`, where the script will parse the py file (for pip install users).

### 📌 \[Recommended\] using Command Line Arguments

Supports directly passing in custom `dataset_id` (compatible with MS and HF) and `dataset_path`, as well as simultaneously passing in multiple custom datasets and their respective sample sizes. The script will automatically preprocess and concatenate the datasets. If a `dataset_id` is passed in, it will default to using the 'default' subset in the dataset_id and set the split to 'train'. If the dataset_id has already been registered, it will use the subsets, split, and preprocessing functions that were passed in during registration. If a `dataset_path` is passed in, it can be specified as a relative path or an absolute path, where the relative path is relative to the current running directory.




```bash
--dataset {dataset_id} {dataset_path}

# Dataset Mixing: the following command takes subset1 and subset2 from dataset_id and samples 20,000 records
--dataset {dataset_name}#20000 {dataset_id}:{subset1}/{subset2}#20000 {dataset_path}#10000
```

The supported file formats for the script include `csv`, `json`, and `jsonl`. You need to ensure that the incoming file conforms to the following dataset formats (only a partial list is provided). All of these formats support the `system` field (it is important to note that if the `system` field is specified in the csv format, it cannot be set to `None` and can only be specified as an empty string. There is no such restriction for the json and jsonl formats). Files in `json` and `jsonl` formats support multi-turn dialogue (`csv` does not support this).


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
system,query,response
00000,11111,22222
00001,aaaaa,bbbbb
00002,AAAAA,BBBBB
```

```jsonl
{"system": "00000", "query": "11111", "response": "22222"}
{"query": "aaaaa", "response": "bbbbb"}
{"query": "AAAAA", "response": "BBBBB"}
```

Multi-Round Dialogue

```jsonl
{"system": "00000", "query": "55555", "response": "66666"}
{"query": "eeeee", "response": "fffff", "history": []}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
```

```json
[{"system": "00000", "query": "55555", "response": "66666"},
{"query": "eeeee", "response": "fffff", "history": []},
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}]
```

**Format 2:**

```jsonl
{"conversations": [{"from": "system", "value": "00000"}, {"from": "user", "value": "11111"}, {"from": "assistant", "value": "22222"}]}
{"conversations": [{"from": "user", "value": "aaaaa"}, {"from": "assistant", "value": "bbbbb"}, {"from": "user", "value": "ccccc"}, {"from": "assistant", "value": "ddddd"}]}
{"conversations": [{"from": "user", "value": "AAAAA"}, {"from": "assistant", "value": "BBBBB"}, {"from": "user", "value": "CCCCC"}, {"from": "assistant", "value": "DDDDD"}]}
```

**Format 3:**

```jsonl
{"messages": [{"role": "system", "content": "00000"}, {"role": "user", "content": "11111"}, {"role": "assistant", "content": "22222"}]}
{"messages": [{"role": "user", "content": "aaaaa"}, {"role": "assistant", "content": "bbbbb"}, {"role": "user", "content": "ccccc"}, {"role": "assistant", "content": "ddddd"}]}
{"messages": [{"role": "user", "content": "AAAAA"}, {"role": "assistant", "content": "BBBBB"}, {"role": "user", "content": "CCCCC"}, {"role": "assistant", "content": "DDDDD"}]}
```

**Format 4:**

```csv
system,instruction,input,output
00000,11111,22222,33333
00001,aaaaa,bbbbb,ccccc
00002,AAAAA,BBBBB,CCCCC
```

**Reinforcement Learning (DPO/ORPO)**

```jsonl
{"query": "11111", "response": "22222", "rejected_response": "33333", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
{"query": "aaaaa", "response": "bbbbb", "rejected_response": "ccccc", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
{"query": "AAAAA", "response": "BBBBB", "rejected_response": "CCCCC", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
```

### Adding dataset_info.json

You can refer to the [builtin dataset_info.json in Swift](https://github.com/modelscope/swift/blob/main/swift/llm/data/dataset_info.json) to expand datasets. You can directly add it in the built-in dataset_info.json, or you can pass in the path to an external dataset_info.json, a JSON string, or a dictionary using `--custom_dataset_info 1.json`.

Adding dataset_id:

```python
# MS
# Usage: `--dataset <dataset_name>`
"<dataset_name>": {
    "dataset_id": "xxx/xxx"
}

# HF
# Usage: `--dataset HF::<dataset_name>` or directly use the `USE_HF` environment variable.
"<dataset_name>": {
    "hf_dataset_id": "xxx/xxx"
}
```

添加dataset\_path:
```python
# You can specify relative and absolute paths. Relative paths are relative to the directory where dataset_info.json is located.
# Usage: `--dataset <dataset_name>`
"<dataset_name>": {
    "dataset_path": "xxx"
}
```

Supported parameters include:

- dataset_id: The corresponding ModelScope dataset_id, default is `None`. The simplest setup requires specifying one of `dataset_id`, `hf_dataset_id`, or `dataset_path`.
- subsets: A list of names of the subsets, default is `[]`, which means using the 'default' subset.
- split: Default is ['train'], usually not necessary to set.
- hf_dataset_id: The corresponding HuggingFace dataset_id, default is `None`.
- dataset_path: Used to specify the local path of the dataset, e.g. 1.jsonl, default is `None`. It can take relative or absolute paths. If using a relative path, it is relative to the directory where the dataset_info.json is located. If dataset_path is set, then dataset_id, subsets, and hf_dataset_id parameters are ignored.
- columns: The default preprocessor used is `SmartPreprocessor`. Specifying this parameter sets it to `RenameColumnsPreprocessor`. You need to rename the columns in the dataset and convert them to the style of **format 1** mentioned above.
- conversations: Specifying this parameter sets the preprocessor to `ConversationsPreprocessor` ('columns' takes priority over 'conversations').
- remove_useless_columns: Specifies whether to remove unnecessary columns (including: 'query', 'response', 'rejected_response', 'system', 'history', 'images'), default is `True`, usually not necessary to set.
- tags: Used to annotate the dataset, default is `[]`, usually not necessary to set.

If the parameters in `dataset_info.json` are not sufficient for your needs, such as adding custom prompts, requiring advanced dataset cleaning, or complex dataset retrieval and preprocessing, you can use the method of registering datasets using functions for data retrieval and preprocessing.

### Registering Datasets

The following is an example of **registering datasets**. The complete py file can be viewed at [custom.py](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/custom.py), and the sh script can be viewed at [custom](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/custom). You can parse the registered content by specifying `--custom_register_path xxx.py`.

```python
from typing import Optional, Tuple

from datasets import Dataset as HfDataset
from modelscope import MsDataset

from swift.llm import get_dataset, register_dataset, get_dataset_from_repo
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


register_dataset(CustomDatasetName.stsb_en, 'huangjintao/stsb', None, _preprocess_stsb, get_dataset_from_repo)


if __name__ == '__main__':
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en],
                                             check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')

```

The `register_dataset` function will register the dataset in the `DATASET_MAPPING`. The parameters of this function are as follows:

- `dataset_name`: Required, representing the name of the dataset, which is also the unique ID of the dataset.
- `dataset_id_or_path`: Required, representing the `dataset_id` on the ModelScope Hub or the local `dataset_dir`.
- `subsets`: List of subsets of the dataset, default is `[]`.
- `split`: Default is ['train'].
- `preprocess_func`: Preprocessing function.
- `get_function`: Default value is `None`. The function to get the dataset. If passed `None`, the decorator approach will be used to register the dataset. If passed a function, the normal approach will be used to register.
   > `get_function` should return `HfDataset` or `Tuple[HfDataset, Optional[HfDataset]]`. If only one dataset is returned, it will be the train_dataset. If two datasets are returned, they will be the train_dataset and val_dataset, respectively. The `get_dataset` function supports obtaining multiple datasets, for example: `get_dataset(['dataset1', 'dataset2'])`. We will concatenate the training and validation parts of each subset and return the merged train_dataset and val_dataset.

   > The `HfDataset` returned by the function needs to follow certain specifications. If you want to do **pre-training**, you only need to include the `response` field, please refer to the `'tigerbot-law-zh'` dataset for details. For **instruction tuning (single-round dialogue)**, the `query` and `response` fields need to be included, representing the user's query and the AI assistant's answer in instruction tuning respectively, please refer to the `'alpaca-zh'` dataset for details. For **multi-round dialogue**, an additional `history` field needs to be added, representing the historical information of the dialogue, please refer to the `'damo-agent-mini-zh'` dataset for details. If each dataset sample has a different `system`, an additional system field needs to be added, you can also refer to the `'damo-agent-mini-zh'` dataset for details.

- `**kwargs`: Other parameters used to annotate the dataset. This parameter generally does not need to be set.

## Custom Models
The following is an example of **custom models**. The complete py file can be viewed at [custom.py](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/custom.py), and the sh script can be viewed at [custom](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/custom). You can parse the registered content by specifying `--custom_register_path xxx.py`.

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
- `revision`: Default is `None`. Used to specify the version number of the model. If `model_id_or_path` is a local model directory, this parameter is not effective. This parameter generally does not need to be set.
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
