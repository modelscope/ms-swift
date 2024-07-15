
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from swift.llm.utils.megatron_utils import (
    MegatronArguments, train_valid_test_datasets_provider,
    model_provider, forward_step
)
from megatron.training import pretrain
from megatron.core.enums import ModelType


# if __name__ == '__main__':

#     args = MegatronArguments(
#         load='/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1',
#         train_data_path=['/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-datasets/alpaca_zh-qwen-train.json'],
#         valid_data_path=['/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-datasets/alpaca_zh-qwen-valid.json'],
#         test_data_path=['/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-datasets/alpaca_zh-qwen-valid.json'],
#         dataset=['alpaca-zh#1000'],
#     )
#     extra_args = args.parse_to_megatron()
#     train_valid_test_datasets_provider.is_distributed = True

#     pretrain(
#         train_valid_test_datasets_provider,
#         model_provider,
#         ModelType.encoder_or_decoder,
#         forward_step,
#         args_defaults=extra_args)


def convert_megatron_to_hf(args_defaults={}):
    sys.path.append('/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen')
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    from hf2mcore_qwen2_dense_and_moe_gqa import convert_checkpoint_from_transformers_to_megatron, check_hf_mg_forward, save_mgmodel
    initialize_megatron(args_defaults=args_defaults)
    args = get_args()

    # if args.convert_checkpoint_from_megatron_to_transformers:
    #     hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path)
    #     mg_model = load_megatron_model(args)
    #     convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
    #     save_hfmodel(args, hf_model)
    # else:
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(args.load)
    mg_model = model_provider()
    convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
    check_hf_mg_forward(hf_model, mg_model, args)
    save_mgmodel(mg_model, args)


if __name__ == '__main__':

    args = MegatronArguments(
        load='/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B',
        save='/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1-new',
        no_async_tensor_model_parallel_allreduce=True,
        no_bias_swiglu_fusion=True,
        bf16=False,
        use_cpu_initialization=True
    )
    extra_args = args.parse_to_megatron()
    convert_megatron_to_hf(extra_args)

