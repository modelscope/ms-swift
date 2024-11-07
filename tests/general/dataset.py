def test_alpaca():
    from swift.llm import load_dataset

    dataset = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'],
                           split_dataset_ratio=1,
                           num_proc=1)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


if __name__ == '__main__':
    test_alpaca()
