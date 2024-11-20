
from typing import Dict, Any, Optional
from datasets import Dataset as HfDataset
from .preprocessor import RowPreprocessor

class KTOPreprocessor(RowPreprocessor):

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        batched_row = dict(batched_row)
        messages = batched_row['messages']
        batch_size = len(messages)
        kl_messages = [messages[-1]] + messages[:-1]

        rejected_response = []
        for i in range(batch_size):
            kl_message = kl_messages[i][-1]
            assert kl_message['role'] == 'assistant'
            rejected_response.append(kl_message['content'])
        batched_row['rejected_response'] = rejected_response

        return batched_row


def get_kl_dataset(dataset: Optional[HfDataset], total_batch_size: int, num_proc: int, seed: Optional[int] = None) -> Optional[HfDataset]:
    # Shift one position to the right in each batch.
    if dataset is None:
        return
    dataset = dataset.shuffle(seed)
    return KTOPreprocessor()(dataset, batch_size=total_batch_size, num_proc=num_proc)
