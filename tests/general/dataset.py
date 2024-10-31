def test_alpaca():
    from swift.llm import load_dataset

    dataset = load_dataset(['alpaca_zh#1000', 'alpaca_en#200'], split_dataset_ratio=1, num_proc=2)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


if __name__ == '__main__':
    test_alpaca()
