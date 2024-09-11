# Copyright (c) Alibaba, Inc. and its affiliates.
from examples.customization.custom import CustomDatasetName, CustomModelType, CustomTemplateType
from swift.llm import (dataset_map, get_dataset, get_model_tokenizer, get_template, print_example)
from swift.utils import get_logger

logger = get_logger()

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