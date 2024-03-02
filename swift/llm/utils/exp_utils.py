import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from queue import Queue
from typing import List, Dict

import torch

from swift.hub.utils.utils import get_cache_dir
from swift.utils.torch_utils import _find_free_port


@dataclass
class Experiment:

    name: str

    cmd: str

    req: Dict = field(default_factory=dict)

    args: Dict = field(default_factory=dict)

    env: Dict = field(default_factory=dict)

    sub_exp: Dict = field(default_factory=dict)

    record: Dict = field(default_factory=dict)

    swift_version: str = None

    requirements_list: str = None

    create_time: float = None

    runtime: Dict = field(default_factory=dict)

    def __init__(self, name, cmd, req, args, sub_exp, **kwargs):
        self.name = name
        self.cmd = cmd
        self.req = req or {}
        self.args = args or {}
        self.sub_exp = sub_exp or {}

    @property
    def priority(self):
        return self.req.get('gpu', 0)

    def to_dict(self):
        _dict = asdict(self)
        _dict.pop('runtime')
        return _dict


class ExpManager:

    def __init__(self):
        self.exps = []

    def run(self, exp: Experiment):
        exp.create_time = time.time()
        runtime = self.build_cmd(exp)
        exp.runtime = runtime
        exp.handler = subprocess.Popen(runtime['running_cmd'])
        self.exps.append(exp)

    def build_cmd(self, exp: Experiment):
        gpu = exp.req.get('gpu', None)
        env = ''
        allocated = []
        if gpu:
            allocated = self.find_free_gpu(gpu)
            assert allocated, 'No free gpu for now!'
            env = f'CUDA_VISIBLE_DEVICES={",".join(allocated)}'
        if exp.req.get('ddp', 1) > 1:
            env += f' NPROC_PER_NODE={exp.req.get("ddp")}'
            env += f' MASTER_PORT={_find_free_port()}'

        if exp.cmd == 'sft':
            from swift.llm import SftArguments
            args = {**exp.args, **exp.sub_exp.get('args', {})}
            sft_args = SftArguments(**args)
            args['output_dir'] = sft_args.output_dir
            args['logging_dir'] = sft_args.logging_dir
            cmd = f'swift sft '
            for key, value in args.items():
                cmd += f' --{key} {value}'
            cmd = env + cmd
        elif exp.cmd == 'dpo':
            from swift.llm import DPOArguments
            args = {**exp.args, **exp.sub_exp.get('args', {})}
            dpo_args = DPOArguments(**args)
            args['output_dir'] = dpo_args.output_dir
            args['logging_dir'] = dpo_args.logging_dir
            cmd = f'swift dpo '
            for key, value in args.items():
                cmd += f' --{key} {value}'
            cmd = env + cmd
        elif exp.cmd == 'export':
            args = {**exp.args, **exp.sub_exp.get('args', {})}
            cmd = f'swift export '
            for key, value in args.items():
                cmd += f' --{key} {value}'
            cmd = env + cmd
        else:
            raise ValueError(f'Unsupported cmd type: {exp.cmd}')
        return {
            'running_cmd': cmd,
            'gpu': allocated,
            'logging_dir': args.get('logging_dir'),
            'output_dir': args.get('output_dir', args.get('ckpt_dir'))
        }

    @staticmethod
    def find_free_gpu(n):
        global exps
        all_gpus = set()
        for exp in exps:
            all_gpus.update(exp.runtime.get('gpu', set()))
        free_gpu = set(range(torch.cuda.device_count())) - all_gpus
        if len(free_gpu) < n:
            return None
        return list(free_gpu)[:n]

    def prepare_exps(self, exp_config: List[str]):
        exps = []
        for config_file in exp_config:
            with open(config_file, 'r') as f:
                content = json.load(f)
                ablation = content.get('ablation', [])
                if not ablation:
                    exps.append(Experiment(**content, sub_exp={}))
                else:
                    for sub_exp in ablation:
                        exps.append(Experiment(**content, sub_exp=sub_exp))
        return exps

    def _get_metric(self, exp: Experiment):
        logging_dir = exp.runtime.get('logging_dir')
        if logging_dir:
            with open(os.path.join(logging_dir, 'logging.jsonl'), 'r') as f:
                for line in f.readlines():
                    if 'model_info' in line:
                        return json.loads(line)
        return None

    def _write_to_file(self, exp: Experiment):
        exp_dir = os.environ.get('EXP_CACHE_DIR', get_cache_dir())
        exp_dir = os.path.join(exp_dir, 'exp_result.jsonl')
        with open(exp_dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(exp.to_dict()) + '\n')

    def _poll(self):
        while True:
            time.sleep(5)
            has_finished = False
            for exp in self.exps:
                rt = exp.handler.poll()
                if rt is None:
                    continue

                has_finished = True
                if rt == 0:
                    all_metric = self._get_metric(exp)
                    exp.record = all_metric
                else:
                    exp.record = {
                        'return_code': rt
                    }
                self._write_to_file(exp)

            if has_finished:
                self.exps = [exp for exp in self.exps if exp.handler.poll() is None]
                break

    def begin(self, exp_config: List[str]):
        exps = self.prepare_exps(exp_config)
        exps.sort(key=lambda e: e.priority)
        exp_queue = Queue(-1)
        for exp in exps:
            exp_queue.put(exp)
        while True:
            try:
                self.run(exp_queue.queue[0])
            except AssertionError:
                break
            else:
                exp_queue.get()
            self._poll()






