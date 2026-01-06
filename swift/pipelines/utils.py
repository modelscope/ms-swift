# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import numpy as np
from datasets import load_from_disk

from swift.dataset import DatasetSyntax, sample_dataset
from swift.plugins import extra_tuners
from swift.template import update_generation_config_eos_token
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()


def prepare_adapter(args, model, adapters=None):
    if args.tuner_backend == 'unsloth':
        if args.model_meta.is_multimodal:
            from unsloth import FastVisionModel as UnslothModel
        else:
            from unsloth import FastLanguageModel as UnslothModel
        UnslothModel.for_inference(model)
        return model
    if args.train_type in extra_tuners:
        tuner = extra_tuners[args.train_type]
    else:
        tuner = Swift
    # compat deploy
    adapters = adapters if adapters is not None else args.adapters
    for adapter in adapters:
        model = tuner.from_pretrained(model, adapter)
    if args.train_type == 'bone':
        # Bone has a problem of float32 matmul with bloat16 in `peft==0.14.0`
        model.to(model.dtype)
    return model


def prepare_model_template(args, **kwargs):
    adapters = kwargs.get('adapters')
    model, processor = args.get_model_processor(**kwargs)
    template = args.get_template(processor)
    if model is not None:
        if template.use_model:
            template.model = model
        model = prepare_adapter(args, model, adapters=adapters)
        update_generation_config_eos_token(model.generation_config, template)
    return model, template


def _select_dataset(dataset, max_length):
    idxs = [
        i for i, length in enumerate(dataset['length'])
        if (max(length) if isinstance(length, list) else length) <= max_length
    ]
    new_dataset = dataset.select(idxs)
    if len(new_dataset) < len(dataset):
        logger.info(f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(new_dataset)}')
    return new_dataset


def get_cached_dataset(args):
    train_datasets, val_datasets = [], []
    random_state = np.random.RandomState(args.data_seed)
    for cached_dataset, datasets in zip([args.cached_dataset, args.cached_val_dataset], [train_datasets, val_datasets]):
        for path in cached_dataset:
            if os.path.exists(path):
                dataset_sample = None
            else:
                path, dataset_sample = DatasetSyntax._safe_split(path, '#', True, 'right')
            dataset = _select_dataset(load_from_disk(path), args.max_length)
            if dataset_sample is not None:
                dataset = sample_dataset(
                    dataset, int(dataset_sample), args.dataset_shuffle, random_state=random_state, shuffle_all=True)
            datasets.append(dataset)
    return train_datasets, val_datasets
