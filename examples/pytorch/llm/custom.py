# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset as HfDataset
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MsDataset
from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import (LoRATM, Template, TemplateType, dataset_map, get_dataset, get_dataset_from_repo,
                       get_model_tokenizer, get_template, print_example, register_dataset, register_model,
                       register_template)
from swift.utils import get_logger

logger = get_logger()


class CustomModelType:
    tigerbot_7b = 'tigerbot-7b'
    tigerbot_13b = 'tigerbot-13b'
    tigerbot_13b_chat = 'tigerbot-13b-chat'


class CustomTemplateType:
    tigerbot = 'tigerbot'


class CustomDatasetName:
    stsb_en = 'stsb-en'


@register_model(CustomModelType.tigerbot_7b, 'TigerResearch/tigerbot-7b-base-v3', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b, 'TigerResearch/tigerbot-13b-base-v2', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(CustomModelType.tigerbot_13b_chat, 'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
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
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template(['{{SYSTEM}}'], ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [], [['eos_token_id']]))


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
    # The Shell script can view `examples/pytorch/llm/scripts/custom`.
    # test dataset
    train_dataset, val_dataset = get_dataset([CustomDatasetName.stsb_en], check_dataset_strategy='warning')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    # test model base
    model, tokenizer = get_model_tokenizer(CustomModelType.tigerbot_13b, use_flash_attn=False)
    # test model chat
    model, tokenizer = get_model_tokenizer(CustomModelType.tigerbot_13b_chat, use_flash_attn=False)
    # test template
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)
