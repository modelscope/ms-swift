import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_mg_model_tokenizer(model_id):
    set_default_ddp_config()
    hf_model, processor = get_model_tokenizer(model_id, torch_dtype=torch.float32)
    megatron_model_meta = get_megatron_model_meta(model_id)
    model_info = processor.model_info
    kwargs = megatron_model_meta.load_config(model_info.config)
    megatron_args = MegatronArguments(**kwargs, seq_length=1, use_cpu_initialization=True)
    patch_megatron(processor)
    extra_args = megatron_args.parse_to_megatron()
    initialize_megatron(args_defaults=extra_args)
    mg_model = megatron_model_meta.get_model_provider()()
    from megatron.training import get_args
    from toolkits.model_checkpoints_convertor.qwen.hf2mcore_qwen2_dense_and_moe_gqa import (
        convert_checkpoint_from_transformers_to_megatron, save_mgmodel, check_hf_mg_forward)
    args = get_args()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    return hf_model, mg_model, processor


def test_align(hf_model, mg_model, template):
    input_ids = template.encode(InferRequest(messages=[{'role': 'user', 'content': '你是谁？'}]))['input_ids']
    input_ids = torch.tensor(input_ids)[None].to('cuda')
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids, -100, True, True, True)
    with torch.inference_mode():
        hf_model.cuda()
        mg_model.cuda()
        hf_logits = hf_model(input_ids).logits
        mg_logits = mg_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    assert (mg_logits - hf_logits).abs().mean().item() < 0.01
    assert (mg_logits - hf_logits).abs().max().item() < 0.1


if __name__ == '__main__':
    import torch
    from megatron.training.initialize import initialize_megatron
    from megatron.training.utils import get_ltor_masks_and_position_ids

    from swift.llm import InferRequest, get_model_tokenizer, get_template, set_default_ddp_config
    from swift.megatron.argument import MegatronArguments
    from swift.megatron.model import get_megatron_model_meta
    from swift.megatron.utils import patch_megatron
    hf_model, mg_model, processor = get_mg_model_tokenizer(model_id)
    template = get_template(hf_model.model_meta.template, processor)
    test_align(hf_model, mg_model, template)
