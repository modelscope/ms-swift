

def test_export_bin_dataset():
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model='Qwen/Qwen2.5-7B-Instruct',
                                dataset='AI-ModelScope/alpaca-gpt4-data-zh',
                                to_bin_dataset=True))
    print()
