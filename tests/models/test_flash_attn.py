from swift.llm import get_model_tokenizer

if __name__ == '__main__':
    # model, tokenizer = get_model_tokenizer('Qwen/Qwen2-7B-Instruct', attn_impl='flash_attn')
    # model, tokenizer = get_model_tokenizer('AIDC-AI/Ovis2-2B', attn_impl='flash_attn')
    # model, tokenizer = get_model_tokenizer('OpenGVLab/InternVL2-2B', attn_impl='flash_attn')
    model, tokenizer = get_model_tokenizer('Shanghai_AI_Laboratory/internlm3-8b-instruct', attn_impl='flash_attn')
    print(model)
