# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Any, Dict, Optional

from datasets import Dataset as HfDataset

from swift.utils import get_dist_setting, get_logger
from ..dataset import RowPreprocessor

logger = get_logger()


class KTOPreprocessor(RowPreprocessor):

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        messages = batched_row['messages']
        batch_size = len(messages)
        kl_messages = [messages[-1]] + messages[:-1]

        kl_response = []
        for i in range(batch_size):
            kl_message = kl_messages[i][-1]
            assert kl_message['role'] == 'assistant'
            kl_response.append(kl_message['content'])
        # The name rejected_response is just for convenience in processing.
        batched_row['rejected_response'] = kl_response

        return batched_row


def _get_kl_dataset(dataset: Optional[HfDataset],
                    total_batch_size: int,
                    num_proc: int,
                    seed: Optional[int] = None) -> Optional[HfDataset]:
    # Shift one position to the right in each batch.
    if dataset is None:
        return
    dataset = dataset.shuffle(seed)
    return KTOPreprocessor()(dataset, batch_size=total_batch_size, num_proc=num_proc)


def prepare_kto_dataset(args, train_dataset, val_dataset):
    world_size = get_dist_setting()[2]
    total_batch_size = (world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)
    if total_batch_size <= 1:
        raise ValueError('Batch size is 1 (too small). KTO will not work properly because the KL term '
                         'will be equivalent to the implied reward.')
    train_dataset = _get_kl_dataset(train_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)
    val_dataset = _get_kl_dataset(val_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)

    label = train_dataset['label']
    num_desirable = max(sum(label), 1)
    num_undesirable = max(len(label) - num_desirable, 1)  # "label" is binary

    if num_desirable != num_undesirable:
        # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
        des_weight_lower_bound = round((num_undesirable * args.undesirable_weight / num_desirable) * 1, 2)
        des_weight_upper_bound = round((num_undesirable * args.undesirable_weight / num_desirable) * 1.33, 2)
        und_weight_lower_bound = round((num_desirable * args.desirable_weight / num_undesirable) / 1.33, 2)
        und_weight_upper_bound = round((num_desirable * args.desirable_weight / num_undesirable) / 1, 2)

        des_weight_in_range = des_weight_lower_bound <= args.desirable_weight <= des_weight_upper_bound
        und_weight_in_range = und_weight_lower_bound <= args.undesirable_weight <= und_weight_upper_bound

        if not (des_weight_in_range or und_weight_in_range):
            logger.info(f'desirable_weight: {args.desirable_weight}, undesirable_weight: {args.undesirable_weight}')
            warnings.warn(
                f"""
        You have different amounts of desirable/positive and undesirable/negative examples but the
        weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
        on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, '{des_weight_upper_bound}]
        or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
        See the documentation on how to optimally set these weights.""", UserWarning)
    return train_dataset, val_dataset
