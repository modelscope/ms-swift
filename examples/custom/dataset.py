from typing import Any, Dict, Optional

from swift.llm import ResponsePreprocessor, register_dataset
from swift.llm.dataset.register import DatasetMeta


class CustomPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row['response']
        prefix_prompt = 'Answer: '
        if response and response.startswith(prefix_prompt):
            response = response[len(prefix_prompt):].strip()
            row['output'] = response
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='YSetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))
