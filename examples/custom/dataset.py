# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class CustomPreprocessor(ResponsePreprocessor):
    prompt = """Task: Based on the given two sentences, provide a similarity score between 0.0 and 5.0.
Sentence 1: {text1}
Sentence 2: {text2}
Similarity score: """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return super().preprocess({
            'query': self.prompt.format(text1=row['text1'], text2=row['text2']),
            'response': f"{row['label']:.1f}"
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/stsb',
        hf_dataset_id='SetFit/stsb',
        preprocess_func=CustomPreprocessor(),
    ))

if __name__ == '__main__':
    dataset = load_dataset(['swift/stsb'])[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')
