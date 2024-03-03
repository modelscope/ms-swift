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

    config_name: str

    cmd: str

    req: Dict = field(default_factory=dict)

    args: Dict = field(default_factory=dict)

    env: Dict = field(default_factory=dict)

    record: Dict = field(default_factory=dict)

    swift_version: str = None

    requirements_list: str = None

    create_time: float = None

    runtime: Dict = field(default_factory=dict)

    input_args: ExpArguments = None

    def __init__(self, name, cmd, req, args, **kwargs):
        self.name = name
        self.cmd = cmd
        self.req = req or {}
        self.args = args or {}

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
        exp_dir = get_cache_dir()
        target_dir = os.path.join(exp_dir, exp.config_name)
        target_file = os.path.join(target_dir, exp.name.replace('/', '-'))
        if os.path.exists(target_file):
            logger.warn(f'Experiment {exp.name} already done, skip')
            with open(target_file, 'r') as f:
                old_exp = Experiment(**json.load(f))
                old_exp.input_args = exp.input_args
                exp = old_exp
        else:
            exp.create_time = time.time()
            runtime = self._build_cmd(exp)
            exp.runtime = runtime
            exp.handler = subprocess.Popen(runtime['running_cmd'])
        self.exps.append(exp)

    def _build_cmd(self, exp: Experiment):
        gpu = exp.req.get('gpu', None)
        env = ''
        allocated = []
        if gpu:
            allocated = self._find_free_gpu(gpu)
            assert allocated, 'No free gpu for now!'
            env = f'CUDA_VISIBLE_DEVICES={",".join(allocated)}'
        if exp.req.get('ddp', 1) > 1:
            env += f' NPROC_PER_NODE={exp.req.get("ddp")}'
            env += f' MASTER_PORT={_find_free_port()}'

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
            cmd = env + cmd
        elif exp.cmd == 'dpo':
            from swift.llm import DPOArguments
            args = exp.args
            dpo_args = DPOArguments(**args)
            args['output_dir'] = dpo_args.output_dir
            args['logging_dir'] = dpo_args.logging_dir
            args['add_output_dir_suffix'] = False
            cmd = f'swift dpo '
            for key, value in args.items():
                cmd += f' --{key} {value}'
            cmd = env + cmd
        elif exp.cmd == 'export':
            args = exp.args
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

    # def name_exists_in_cache(self, name):
    #     exp_dir = get_cache_dir()
    #     return os.path.exists(os.path.join(exp_dir, name))
    #
    # def name_exists_in_hub(self, name):
    #     return check_model_is_id(name)

    # def check_name_available(self, name, args: ExpArguments):
    #     if self.name_exists_in_cache(name):
    #         raise AssertionError(f'Experiment name {name} exists in local cache.')
    #     if args.push_to_hub and self.name_exists_in_hub(name):
    #         raise AssertionError(f'Experiment name {name} exists in model hub.')

    def _find_free_gpu(self, n):
        all_gpus = set()
        for exp in self.exps:
            all_gpus.update(exp.runtime.get('gpu', set()))
        free_gpu = set(range(torch.cuda.device_count())) - all_gpus
        if len(free_gpu) < n:
            return None
        return list(free_gpu)[:n]

    @staticmethod
    def _merge_sub_config(config, sub_config):
        if 'req' in config:
            config['req'].update(config, sub_config.get('req', {}))
        if 'args' in config:
            config['args'].update(config, sub_config.get('args', {}))
        if 'env' in config:
            config['env'].update(config, sub_config.get('env', {}))
        return config

    def prepare_experiments(self, args: ExpArguments):
        experiments = []
        for config_file in args.config:
            config_file = self._store_config(config_file, args)
            with open(config_file, 'r') as f:
                content = json.load(f)
                ablation = content.get('ablation', [])
                config_name = content['name']
                if not ablation:
                    new_name = content['name'] + '-' + args.name
                    # self.check_name_available(new_name, args)
                    content['name'] = new_name
                    experiments.append(Experiment(**content, config_name=config_name, input_args=args))
                else:
                    for sub_exp in ablation:
                        new_name = content['name'] + '-' + sub_exp.name + '-' + args.name
                        # self.check_name_available(new_name, args)
                        content = self._merge_sub_config(content, sub_exp)
                        content['name'] = new_name
                        experiments.append(Experiment(**content, config_name=config_name, input_args=args))
        return experiments

    @staticmethod
    def _get_metric(exp: Experiment):
        logging_dir = exp.runtime.get('logging_dir')
        if logging_dir:
            with open(os.path.join(logging_dir, 'logging.jsonl'), 'r') as f:
                for line in f.readlines():
                    if 'model_info' in line:
                        return json.loads(line)
        return None

    @staticmethod
    def write_record(exp: Experiment):
        target_dir = os.path.join(get_cache_dir(), exp.config_name)
        file = os.path.join(target_dir, exp.name.replace('/', '-'))
        with open(file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(exp.to_dict()) + '\n')
        if exp.input_args.push_to_hub and exp.input_args.update_exist_config:
            push_to_hub(exp.config_name, target_dir, private=exp.input_args.private)

    @staticmethod
    def _store_config(config_file, args: ExpArguments):
        exp_dir = get_cache_dir()
        with open(config_file, 'r') as f:
            name = json.load(f)['name']
        target_folder = os.path.join(exp_dir, name)
        target_file = os.path.exists(os.path.join(target_folder, 'experiment', 'config.json'))
        if target_file == config_file:
            return config_file

        target_file = os.path.join(target_folder, 'experiment', 'config.json')
        if os.path.isfile(target_file):
            with open(target_file, 'r') as tf:
                _target = json.load(tf)
            with open(config_file, 'r') as f:
                _source = json.load(f)

            def ordered(obj):
                if isinstance(obj, dict):
                    return sorted((k, ordered(v)) for k, v in obj.items())
                if isinstance(obj, list):
                    return sorted(ordered(x) for x in obj)
                else:
                    return obj

            if ordered(_target) != ordered(_source):
                shutil.rmtree(target_folder)
        os.makedirs(os.path.join(target_folder, 'experiment'), exist_ok=True)
        shutil.copy(config_file, target_file)
        if args.push_to_hub and args.update_exist_config:
            push_to_hub(name, target_folder, private=args.private)
        return target_file

    def _poll(self):
        while True:
            time.sleep(5)

            has_finished = False
            for exp in self.exps:
                if exp.record:
                    if exp.input_args.push_to_hub and exp.input_args.update_exist_config:
                        target_dir = os.path.join(get_cache_dir(), exp.config_name)
                        push_to_hub(exp.config_name, target_dir, private=exp.input_args.private)
                else:
                    rt = exp.handler.poll()
                    if rt is None:
                        continue

                    has_finished = True
                    if rt == 0:
                        all_metric = self._get_metric(exp)
                        exp.record['return_code'] = rt
                        if all_metric:
                            exp.record.update(all_metric)
                        self.write_record(exp)
                        checkpoint_path = exp.record.get('best_model_checkpoint')
                        if checkpoint_path and exp.input_args.push_to_hub:
                            upload_handler = push_to_hub_async(exp.name, checkpoint_path,
                                                               private=exp.input_args.private)

                            def upload_callback(future: Future[bool]):
                                if future.result():
                                    exp.record['best_model_checkpoint_id'] = exp.name
                                    self.write_record(exp)
                                else:
                                    logger.error(f'Uploading {exp.name} failed')

                            upload_handler.add_done_callback(upload_callback)
                    else:
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






