from typing import Any, Dict, Optional

from swift.llm import AlpacaPreprocessor, register_dataset
from swift.llm.dataset.register import DatasetMeta


class CustomPreprocessor(AlpacaPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = row['response']
        prefix_prompt = 'Answer: '
        if response and response.startswith(prefix_prompt):
            response = response[len(prefix_prompt):].strip()
            row['output'] = response
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LongAlpaca-12k',
        hf_dataset_id='Yukang/LongAlpaca-12k',
        preprocess_func=CustomPreprocessor(),
    ))
