



def test_qwen2():
    from swift.llm import get_model_tokenizer

    model, tokenizer = get_model_tokenizer('qwen/Qwen2-7B-Instruct', torch.float32, )


    