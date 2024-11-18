# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import ExportArguments, SwiftPipeline, merge_lora


class SwiftMergeLoRA(SwiftPipeline):
    args_class = ExportArguments
    args: args_class

    def run(self):
        merge_lora(self.args)


if __name__ == '__main__':
    SwiftMergeLoRA().main()
