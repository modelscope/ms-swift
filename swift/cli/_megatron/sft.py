# Copyright (c) Alibaba, Inc. and its affiliates.
import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    from swift.megatron import megatron_sft_main
    megatron_sft_main()
