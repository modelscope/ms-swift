# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import time
from typing import List, Union

from swift.llm import SamplingArguments, SwiftPipeline, load_dataset
from swift.utils import get_logger

logger = get_logger()


class SwiftSampling(SwiftPipeline):
    args_class = SamplingArguments
    args: args_class

    def __init__(self, args: Union[List[str], SamplingArguments, None] = None) -> None:
        super().__init__(args)
        self.args.save_args()
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.cur_piece = 0
        self.total_piece = 1

        if self.args.data_range:
            self.cur_piece, self.total_piece = self.args.data_range

        if self.args.sampler_type == 'sample':
            from swift.llm.sampling.vanilla_sampler import VanillaSampler
            self.sampler = VanillaSampler(self.args)
        elif self.args.sampler_type == 'mcts':
            from swift.llm.sampling.mcts import MctsSampler
            self.sampler = MctsSampler(self.args)
        elif self.args.sampler_type == 'distill':
            from swift.llm.sampling.distill_sampler import DistillSampler
            self.sampler = DistillSampler(self.args)
        else:
            raise ValueError(f'Unsupported sampler type: {self.args.sampler_type}')

    def _get_dataset(self):
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        sampling_dataset, _ = load_dataset(args.dataset, split_dataset_ratio=0., **dataset_kwargs)
        logger.info(f'Sampling_dataset: {sampling_dataset}')
        dataset_len = len(sampling_dataset)
        piece_len = dataset_len // self.total_piece
        sampling_dataset = sampling_dataset.select(range(piece_len * self.cur_piece, piece_len * (self.cur_piece + 1)))
        return sampling_dataset

    def run(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        iter_file = os.path.join(self.args.output_dir, self.args.output_file)
        tmp_file = os.path.join(self.args.output_dir, self.args.output_file + '.tmp')
        if os.path.exists(iter_file) and not self.args.override_exist_file:
            return
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        dataset = self._get_dataset()
        dataset_len = len(dataset)
        total_iters = int(dataset_len // self.args.num_sampling_per_gpu_batch_size)
        if self.args.num_sampling_per_gpu_batches is None or self.args.num_sampling_per_gpu_batches > total_iters:
            self.args.num_sampling_per_gpu_batches = total_iters

        with open(tmp_file, 'w') as f:
            for _index in range(self.args.num_sampling_per_gpu_batches):
                logger.info(f' Sampling index:{_index}')
                slices = dataset[self.args.num_sampling_per_gpu_batch_size
                                 * _index:self.args.num_sampling_per_gpu_batch_size * (_index + 1)]
                slices = self.sampler.truncate_input(slices)
                generated = self.sampler.do_sample(slices)
                f.writelines(generated)
        if os.path.exists(iter_file):
            shutil.move(iter_file, iter_file + '.' + str(int(time.time())))
        shutil.move(tmp_file, iter_file)
        logger.info(f'Sample file {iter_file} generated.')


def sampling_main(args: Union[List[str], SamplingArguments, None] = None):
    return SwiftSampling(args).main()
