import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_PORT'] = '29560'


def get_mg_model_tokenizer(model_id):
    from megatron.training.initialize import initialize_megatron
    set_default_ddp_config()
    hf_model, processor = get_model_tokenizer(model_id, torch_dtype=torch.float32)
    megatron_model_meta = get_megatron_model_meta(processor.model_meta.model_type)
    model_info = processor.model_info
    kwargs = megatron_model_meta.convert_hf_config(model_info.config)
    megatron_args = MegatronArguments(
        **kwargs, seq_length=1, use_cpu_initialization=True, no_initialization=True, torch_dtype=torch.float32)
    extra_args_provider = megatron_model_meta.extra_args_provider
    patch_megatron_tokenizer(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args, extra_args_provider=extra_args_provider)
    mg_model = megatron_model_meta.model_provider()
    megatron_model_meta.convert_hf2mcore(hf_model, mg_model)
    return hf_model, mg_model, processor


def test_bf16_fp32():
    hf_model_fp32, processor = get_model_tokenizer(model_id, torch_dtype=torch.float32)
    hf_model_bf16, processor = get_model_tokenizer(model_id, torch_dtype=torch.bfloat16)
    template = get_template(hf_model_fp32.model_meta.template, processor)
    input_ids = template.encode(InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]))['input_ids']
    input_ids = torch.tensor(input_ids)[None].to('cuda')
    with torch.inference_mode():
        hf_logits_fp32 = hf_model_fp32(input_ids).logits
        hf_logits_bf16 = hf_model_bf16(input_ids).logits
    mean_diff = (hf_logits_fp32 - hf_logits_bf16).abs().mean().item()
    max_diff = (hf_logits_fp32 - hf_logits_bf16).abs().max().item()
    # mean_diff: 0.13342587649822235, max_diff: 7.1983513832092285
    print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')


def test_align(hf_model, mg_model, processor):
    from megatron.training.utils import get_ltor_masks_and_position_ids
    template = get_template(hf_model.model_meta.template, processor)
    input_ids = template.encode(InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]))['input_ids']
    input_ids = torch.tensor(input_ids)[None].to('cuda')
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    with torch.inference_mode():
        hf_model.cuda()
        mg_model.cuda()
        hf_logits = hf_model(input_ids).logits
        mg_logits = mg_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    mean_diff = (mg_logits - hf_logits).abs().mean().item()
    max_diff = (mg_logits - hf_logits).abs().max().item()
    print(f'mean_diff: {mean_diff}, max_diff: {max_diff}')


model_id = 'Qwen/Qwen2-7B-Instruct'

if __name__ == '__main__':
    import torch
    from swift.llm import InferRequest, get_model_tokenizer, get_template
    from swift.utils import set_default_ddp_config
    from swift.megatron.argument import MegatronArguments
    from swift.megatron.model import get_megatron_model_meta
    from swift.megatron.utils import patch_megatron_tokenizer
    # test_bf16_fp32()
    hf_model, mg_model, processor = get_mg_model_tokenizer(model_id)
    test_align(hf_model, mg_model, processor)
