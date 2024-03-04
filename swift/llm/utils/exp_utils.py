import json
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, asdict
from queue import Queue
from typing import Dict
from copy import deepcopy

import torch

from swift import push_to_hub, push_to_hub_async
from swift.hub.utils.utils import get_cache_dir
from swift.llm import ExpArguments
from swift.utils import get_logger
from swift.utils.torch_utils import _find_free_port

logger = get_logger()


@dataclass
class Experiment:

    name: str

    cmd: str

    requirements: Dict = field(default_factory=dict)

    args: Dict = field(default_factory=dict)

    env: Dict = field(default_factory=dict)

    record: Dict = field(default_factory=dict)

    create_time: float = None

    runtime: Dict = field(default_factory=dict)

    input_args: ExpArguments = None

    def __init__(self, name, cmd, requirements=None, args=None, input_args=None, **kwargs):
        self.name = name
        self.cmd = cmd
        self.requirements = requirements or {}
        self.args = args or {}
        self.record = {}
        self.env = {}
        self.runtime = {}
        self.input_args = input_args

    @property
    def priority(self):
        return self.requirements.get('gpu', 0)

    def to_dict(self):
        _dict = asdict(self)
        _dict.pop('runtime')
        _dict.pop('input_args')
        return _dict


class ExpManager:

    RESULT_FILE = 'result.jsonl'

    def __init__(self):
        self.exps = []

    def run(self, exp: Experiment):
        if os.path.exists(os.path.join(exp.input_args.save_dir, exp.name)):
            logger.warn(f'Experiment {exp.name} already done, skip')
        else:
            exp.create_time = time.time()
            runtime = self._build_cmd(exp)
            exp.runtime = runtime
            envs = runtime.get('env', {})
            envs.update(os.environ)
            exp.handler = subprocess.Popen(runtime['running_cmd'], env=envs, shell=True)
        self.exps.append(exp)

    def _build_cmd(self, exp: Experiment):
        gpu = exp.requirements.get('gpu', None)
        env = {}
        allocated = []
        if gpu:
            allocated = self._find_free_gpu(int(gpu))
            assert allocated, 'No free gpu for now!'
            allocated = [str(gpu) for gpu in allocated]
            env['CUDA_VISIBLE_DEVICES'] = ",".join(allocated)
        if int(exp.requirements.get('ddp', 1)) > 1:
            env['NPROC_PER_NODE'] = exp.requirements.get("ddp")
            env['MASTER_PORT'] = str(_find_free_port())

        if exp.cmd == 'sft':
            from swift.llm import SftArguments
            args = exp.args
            sft_args = SftArguments(**args)
            args['output_dir'] = sft_args.output_dir
            args['logging_dir'] = sft_args.logging_dir
            args['add_output_dir_suffix'] = False
            cmd = f'swift sft '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        elif exp.cmd == 'dpo':
            from swift.llm import DPOArguments
            args = exp.args
            dpo_args = DPOArguments(**args)
            args['output_dir'] = dpo_args.output_dir
            args['logging_dir'] = dpo_args.logging_dir
            args['add_output_dir_suffix'] = False
            cmd = f' swift dpo '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        elif exp.cmd == 'export':
            args = exp.args
            cmd = f' swift export '
            for key, value in args.items():
                cmd += f' --{key} {value}'
        else:
            raise ValueError(f'Unsupported cmd type: {exp.cmd}')
        return {
            'running_cmd': cmd,
            'gpu': allocated,
            'env': env,
            'logging_dir': args.get('logging_dir'),
            'output_dir': args.get('output_dir', args.get('ckpt_dir'))
        }

    def _find_free_gpu(self, n):
        all_gpus = set()
        for exp in self.exps:
            all_gpus.update(exp.runtime.get('gpu', set()))
        free_gpu = set(range(torch.cuda.device_count())) - all_gpus
        if len(free_gpu) < n:
            return None
        return list(free_gpu)[:n]

    def prepare_experiments(self, args: ExpArguments):
        experiments = []
        for config_file in args.config:
            with open(config_file, 'r') as f:
                content = json.load(f)
                exps = content['experiment']
                for exp in exps:
                    name = exp['name']
                    cmd = content['cmd']
                    args = content['args']
                    env = content['env']
                    requirements = content['requirements']
                    if 'args' in exp:
                        args.update(exp['args'])
                    if 'requirements' in exp:
                        requirements.update(exp['requirements'])
                    if 'env' in exp:
                        env.update(exp['env'])
                    experiments.append(Experiment(name=name, cmd=cmd, args=args, env=env, requirements=requirements,
                                                  input_args=args))
        return experiments

    @staticmethod
    def _get_metric(exp: Experiment):
        logging_dir = exp.runtime.get('logging_dir')
        if logging_dir:
            with open(os.path.join(logging_dir, '..', 'logging.jsonl'), 'r') as f:
                for line in f.readlines():
                    if 'model_info' in line:
                        return json.loads(line)
        return None

    @staticmethod
    def write_record(exp: Experiment):
        target_dir = exp.input_args.save_dir
        file = os.path.join(target_dir, exp.name + '.json')
        with open(file, 'w', encoding='utf-8') as f:
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
                    if all_metric:
                        exp.record.update(all_metric)
                    self.write_record(exp)
                logger.error(f'Running {exp.name} finished with return code: {rt}')

            if has_finished:
                self.exps = [exp for exp in self.exps if not exp.record and exp.handler.poll() is None]
                break

    def begin(self, args: ExpArguments):
        exps = self.prepare_experiments(args)
        exps.sort(key=lambda e: e.priority)
        exp_queue = Queue(-1)
        for exp in exps:
            exp_queue.put(exp)

        while not exp_queue.empty() or len(self.exps) > 0:
            while not exp_queue.empty():
                try:
                    self.run(exp_queue.queue[0])
                except AssertionError:
                    break
                else:
                    exp_queue.get()
            self._poll()






