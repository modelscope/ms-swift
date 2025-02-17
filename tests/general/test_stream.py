from swift.llm import load_dataset


def test_local_dataset():
    # please use git clone
    from swift.llm import git_clone_github
    model_dir = git_clone_github('https://www.modelscope.cn/datasets/swift/swift-sft-mixture.git')
    dataset = load_dataset(datasets=[f'{model_dir}:firefly'], streaming=True)[0]
    print(next(iter(dataset)))


def test_hub_dataset():
    local_dataset = 'swift/swift-sft-mixture:firefly'
    dataset = load_dataset(datasets=[local_dataset], streaming=True)[0]
    print(next(iter(dataset)))


if __name__ == '__main__':
    test_local_dataset()
    # test_hub_dataset()
