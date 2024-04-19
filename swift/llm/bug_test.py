from utils import get_dataset

train_dataset, val_dataset = get_dataset(
    ['ms-bench'],
    0.01,
    check_dataset_strategy='warning')
