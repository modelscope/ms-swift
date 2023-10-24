# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from typing import Any, Callable, Dict, List, Optional

from modelscope import (AutoModelForCausalLM, AutoTokenizer, MsDataset,
                        snapshot_download)
from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import (HfDataset, History, LoRATM, Template,
                       concatenate_datasets, dataset_map, get_dataset,
                       get_model_tokenizer, get_model_tokenizer_from_repo,
                       get_template, preprocess_conversations,
                       register_dataset, register_model, register_template)
from swift.utils import get_logger, print_example

logger = get_logger()


class CustomModelType:
    tigerbot_13b_chat = 'tigerbot-13b-chat'


class CustomTemplateType:
    tigerbot = 'tigerbot'


class CustomDatasetName:
    agent_instruct_all_en = 'agent-instruct-all-en'


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
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)


# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template([], ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [],
             [['eos_token_id']]))

_agent_instruct_subset_list = [
    'alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'
]


@register_dataset(
    CustomDatasetName.agent_instruct_all_en,
    task='chat',
    function_kwargs={'subset_name_list': _agent_instruct_subset_list})
def get_agent_instruct_dataset(subset_name_list: List[str]) -> HfDataset:
    dataset_list: List[HfDataset] = []
    for subset_name in subset_name_list:
        dataset: HfDataset = MsDataset.load(
            'huangjintao/AgentInstruct_copy',
            subset_name=subset_name,
            split='train').to_hf_dataset()
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)

    def repair_conversations(s: str) -> str:
        s = s.replace('}\n {', '},\n {')
        return ast.literal_eval(s)

    return preprocess_conversations(
        dataset, 'human', 'gpt', repair_conversations=repair_conversations)


if __name__ == '__main__':
    # test
    from swift.llm import DatasetName
    train_dataset, _ = get_dataset([CustomDatasetName.agent_instruct_all_en],
                                   0.)
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b_chat, use_flash_attn=True)
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)
