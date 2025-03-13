# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import time
from typing import List, Union

import json

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
        resume_file = os.path.join(self.args.output_dir, self.args.output_file + '.resume')
        tmp_file = os.path.join(self.args.output_dir, self.args.output_file + '.tmp')
        ckpt_state_file = os.path.join(self.args.output_dir, 'ckpt_state.json')
        if os.path.exists(iter_file) and not self.args.override_exist_file:
            return

        index_resume = -1
        write_mode = 'w'
        if self.args.resume:
            write_mode = 'a'
            if os.path.exists(resume_file):
                shutil.copyfile(resume_file, tmp_file)

            if os.path.exists(ckpt_state_file):
                with open(ckpt_state_file, 'r') as ckpt_state:
                    data = json.load(ckpt_state)
                    index_resume = data.get('index', -1)
                    logger.info(f'Loaded index_resume: {index_resume}')
        else:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

        dataset = self._get_dataset()
        dataset_len = len(dataset)
        total_iters = int(dataset_len // self.args.num_sampling_per_gpu_batch_size)

        if self.args.num_sampling_per_gpu_batches is None or self.args.num_sampling_per_gpu_batches > total_iters:
            self.args.num_sampling_per_gpu_batches = total_iters

        with open(tmp_file, write_mode) as f:
            for _index in range(self.args.num_sampling_per_gpu_batches):
                if _index <= index_resume:
                    continue
                logger.info(f' Sampling index:{_index}')
                slices = dataset[self.args.num_sampling_per_gpu_batch_size
                                 * _index:self.args.num_sampling_per_gpu_batch_size * (_index + 1)]
                slices = self.sampler.truncate_input(slices)
                generated = self.sampler.do_sample(slices)
                f.writelines(generated)
                f.flush()
                shutil.copy(tmp_file, resume_file)
                with open(ckpt_state_file, 'w') as ckpt_state:
                    json.dump({'index': _index}, ckpt_state)

        if os.path.exists(iter_file):
            shutil.move(iter_file, iter_file + '.' + str(int(time.time())))
        shutil.move(resume_file, iter_file)
        logger.info(f'Sample file {iter_file} generated.')


def sampling_main(args: Union[List[str], SamplingArguments, None] = None):
    return SwiftSampling(args).main()
