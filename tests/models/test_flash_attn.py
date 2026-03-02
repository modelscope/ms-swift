from swift.model import get_model_processor

if __name__ == '__main__':
    # model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', attn_impl='flash_attn')
    # model, tokenizer = get_model_processor('AIDC-AI/Ovis2-2B', attn_impl='flash_attn')
    # model, tokenizer = get_model_processor('OpenGVLab/InternVL2-2B', attn_impl='flash_attn')
    model, tokenizer = get_model_processor('Shanghai_AI_Laboratory/internlm3-8b-instruct', attn_impl='flash_attn')
    print(model)
