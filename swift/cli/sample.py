# Copyright (c) ModelScope Contributors. All rights reserved.

if __name__ == '__main__':
    from swift.utils.ray_utils import try_init_ray
    try_init_ray()
    from swift.pipelines import sampling_main
    sampling_main()
