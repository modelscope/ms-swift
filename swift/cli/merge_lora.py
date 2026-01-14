# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.arguments import ExportArguments
from swift.pipelines import SwiftPipeline, merge_lora


class SwiftMergeLoRA(SwiftPipeline):
    args_class = ExportArguments
    args: args_class

    def run(self):
        merge_lora(self.args)


if __name__ == '__main__':
    SwiftMergeLoRA().main()
