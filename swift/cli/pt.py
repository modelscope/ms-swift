# Copyright (c) Alibaba, Inc. and its affiliates.

if __name__ == '__main__':
    from swift.cli.utils import fix_ppu
    fix_ppu()
    from swift.llm import pt_main
    pt_main()
