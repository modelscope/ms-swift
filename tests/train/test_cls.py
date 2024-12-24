from swift.llm import TrainArguments, sft_main


def test_llm():
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            num_labels=2,
            #  dataset=['simpleai/HC3-Chinese:baike_cls#1000', ],
            dataset=['simpleai/HC3:finance_cls#1000'],
            use_chat_template=False))


if __name__ == '__main__':
    test_llm()
