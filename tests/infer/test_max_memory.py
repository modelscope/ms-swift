from swift.llm import InferArguments, infer_main


def test_max_memory():
    infer_main(
        InferArguments(model='Qwen/Qwen2.5-7B-Instruct', max_memory='{0: "50GB", 1: "5GB"}', device_map='sequential'))


if __name__ == '__main__':
    test_max_memory()
