# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
from typing import Any, Dict

from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import (ConversationsPreprocessor, LoRATM, Template,
                       dataset_map, get_dataset, get_dataset_from_repo,
                       get_model_tokenizer, get_model_tokenizer_from_repo,
                       get_template, print_example, register_dataset,
                       register_model, register_template)
from swift.utils import get_logger

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


def repair_conversations_agent_instruct(s: str) -> str:
    s = s.replace('}\n {', '},\n {')
    return ast.literal_eval(s)


register_dataset(
    CustomDatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy',
    [(subset, 'train') for subset in _agent_instruct_subset_list],
    None,
    ConversationsPreprocessor(
        'human',
        'gpt',
        repair_conversations=repair_conversations_agent_instruct),
    get_dataset_from_repo,
    task='chat')

if __name__ == '__main__':
    # The Shell script can view `scripts/custom/tigerbot_13b_chat`.
    # test
    train_dataset, _ = get_dataset([CustomDatasetName.agent_instruct_all_en],
                                   0.)
    model, tokenizer = get_model_tokenizer(
        CustomModelType.tigerbot_13b_chat, use_flash_attn=True)
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    train_dataset = dataset_map(train_dataset, template.encode)
    print_example(train_dataset[0], tokenizer)
