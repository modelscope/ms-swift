def test_app():
    from swift.llm import app_main, AppArguments
    app_main(AppArguments(model='Qwen/Qwen2.5-0.5B-Instruct'))


if __name__ == '__main__':
    test_app()
