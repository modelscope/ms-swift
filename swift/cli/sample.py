# Copyright (c) Alibaba, Inc. and its affiliates.

if __name__ == '__main__':
    from swift.ray import try_init_ray
    try_init_ray()
    from swift.llm.sampling import sampling_main
    sampling_main()
