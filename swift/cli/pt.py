# Copyright (c) Alibaba, Inc. and its affiliates.

if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.llm import pt_main
    pt_main()
