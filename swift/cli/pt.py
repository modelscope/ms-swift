# Copyright (c) ModelScope Contributors. All rights reserved.

if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import pretrain_main
    pretrain_main()
