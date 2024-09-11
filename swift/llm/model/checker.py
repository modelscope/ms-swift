

def check_awq_ext() -> None:
    try:
        from awq.utils.packing_utils import dequantize_gemm
        import awq_ext  # with CUDA kernels (AutoAWQ_kernels)
    except ImportError as e:
        raise ImportError('You are training awq models, remember installing awq_ext by '
                          '`git clone https://github.com/casper-hansen/AutoAWQ_kernels '
                          '&& cd AutoAWQ_kernels && pip install -e .`') from e
