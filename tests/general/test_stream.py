from swift.llm import load_dataset


def test_local_dataset():
    # please use git clone
    local_dataset = '/mnt/nas2/huangjintao.hjt/work/datasets/swift-sft-mixture:firefly#100'
    dataset = load_dataset(datasets=[local_dataset], streaming=True)[0]
    for i, x in enumerate(dataset):
        pass
    print(i, x)


def test_hub_dataset():
    local_dataset = 'swift/swift-sft-mixture:firefly'
    dataset = load_dataset(datasets=[local_dataset], streaming=True)[0]
    print(next(iter(dataset)))


if __name__ == '__main__':
    test_local_dataset()
    # test_hub_dataset()
