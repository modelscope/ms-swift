# merge-lora
CUDA_VISIBLE_DEVICES=0 swift export \
    --adapters swift/test_bert \
    --output_dir output/swift_test_bert_merged \
    --merge_lora true

# gptq quantize
CUDA_VISIBLE_DEVICES=0 swift export \
    --model output/swift_test_bert_merged \
    --load_data_args true \
    --output_dir output/swift_test_bert_gptq_int4 \
    --quant_bits 4 \
    --quant_method gptq \
    --max_length 512

# infer
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model output/swift_test_bert_gptq_int4
