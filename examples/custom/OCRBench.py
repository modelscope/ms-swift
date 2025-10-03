
from typing import Any, Dict, Optional
from swift.llm import (load_dataset, register_dataset, DatasetMeta)
from swift.llm.dataset import ResponsePreprocessor



class OCRBenchPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import os
        row = super().preprocess(row)
        image = row['image_path']
        if not image:
            return
        image = os.path.join(self.prefix_path, image)
        if not os.path.exists(image):
            return
        row['images'] = [image]
        return row

    def prepare_dataset(self, dataset):
        "local dataset"
        
        self.prefix_path = "/datasets/OCRBench/OCRBench_Images/"
        return super().prepare_dataset(dataset)


register_dataset(
    DatasetMeta(
        dataset_path='/datasets/OCRBench/OCRBench.json',
        preprocess_func=OCRBenchPreprocessor(),
        
    ))




if __name__ == '__main__':
    # The Shell script can view `examples/pytorch/llm/scripts/custom`.
    # test dataset
    train_dataset, val_dataset = load_dataset('/datasets/OCRBench/OCRBench.json')
    print(f'train_dataset: {train_dataset}')
    print(f'val_dataset: {val_dataset}')
    
