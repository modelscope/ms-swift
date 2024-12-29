def test_llm():
    from swift.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2.5-0.5B-Instruct'))


def test_lora():
    from swift.llm import app_main, AppArguments
    app_main(AppArguments(adapters='swift/test_lora', lang='en', studio_title='小黄'))


def test_mllm():
    from swift.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2-VL-7B-Instruct', stream=True))


def test_audio():
    from swift.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2-Audio-7B-Instruct', stream=True))


if __name__ == '__main__':
    test_audio()
