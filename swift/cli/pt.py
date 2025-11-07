# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import pt_main

if __name__ == '__main__':
    from swift.cli.utils import fix_ppu
    fix_ppu()
    from swift.ray import try_init_ray
    try_init_ray()
    from swift.llm import pt_main
    pt_main()
