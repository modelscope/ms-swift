# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import glob
import os
from contextlib import nullcontext
from evalscope.constants import EvalBackend, EvalType
from evalscope.run import TaskConfig, run_task
from evalscope.summarizer import Summarizer
from typing import List, Optional, Union

from swift.arguments import EvalArguments
from swift.dataset import MediaResource
from swift.utils import append_to_jsonl, gc_collect, get_logger
from ..base import SwiftPipeline
from ..infer import run_deploy

logger = get_logger()


class SwiftEval(SwiftPipeline):
    args_class = EvalArguments
    args: args_class

    def run(self):
        args = self.args
        eval_report = {}
        if args.eval_ckpts:
            eval_report[args.eval_backend] = self.run_ckpt_sweep()
        elif args.eval_offline:
            result = self.get_task_result(self.get_offline_task_cfg())
            eval_report[args.eval_backend] = result
        else:
            deploy_context = nullcontext() if args.eval_url else run_deploy(args, return_url=True)
            with deploy_context as base_url:
                base_url = args.eval_url or base_url

                task_cfg = self.get_task_cfg(args.eval_dataset, args.eval_backend, base_url)
                result = self.get_task_result(task_cfg)
                eval_report[args.eval_backend] = result

        eval_report.update({
            'time': args.time,
            'model': args.model,
            'adapters': args.adapters,
            'result_path': args.result_path,
            'eval_output_dir': args.eval_output_dir,
            'eval_limit': args.eval_limit
        })

        if args.result_jsonl:
            append_to_jsonl(args.result_jsonl, eval_report)
            logger.info(f'The eval result have been saved to result_jsonl: `{args.result_jsonl}`.')
        return eval_report

    def get_task_result(self, task_cfg: TaskConfig):
        run_task(task_cfg=task_cfg)
        reports = Summarizer.get_report_from_cfg(task_cfg=task_cfg)
        result = {}
        if task_cfg.eval_backend == EvalBackend.OPEN_COMPASS:
            for report in reports:
                if report[self.args.model_suffix] != '-':
                    result[report['dataset']] = {report['metric']: report[self.args.model_suffix]}
        elif task_cfg.eval_backend == EvalBackend.VLM_EVAL_KIT:
            for report in reports:
                splited_key = next(iter(report)).rsplit('_', 2)
                if len(splited_key) == 3:
                    _, dataset, metric = splited_key
                else:
                    dataset, metric = '-', '-'
                result[dataset] = {metric: list(report.values())[0]}
        else:
            result = reports
        return result

    def get_offline_task_cfg(self, args=None):
        """Evaluate the model in-process instead of calling a deployed server over HTTP.

        Uses the same `swift_custom` engine the training loop already uses, so both paths run the
        same evaluation implementation -- this one just skips the deploy and its engine startup.
        """
        args = args or self.args
        if args.eval_backend != EvalBackend.NATIVE:
            raise ValueError('eval_offline is only supported by the Native backend, '
                             f'got eval_backend: {args.eval_backend}')
        from ..utils import prepare_model_template
        from .utils import EvalModel

        model, template = prepare_model_template(args)
        template.packing = False
        template.padding_free = False
        model.eval()
        return TaskConfig(
            model=EvalModel(
                model_name=args.model_suffix or str(args.model),
                model=model,
                template=template,
                max_batch_size=args.eval_num_proc,
            ),
            eval_type='swift_custom',
            datasets=args.eval_dataset,
            work_dir=os.path.join(args.eval_output_dir, 'offline'),
            limit=args.eval_limit,
            eval_batch_size=args.eval_num_proc,
            dataset_args=args.eval_dataset_args,
            generation_config=args.eval_generation_config or {'max_tokens': 512},
            **args.extra_eval_args)

    def run_ckpt_sweep(self):
        """Evaluate several checkpoints one after another and report them side by side.

        Because the model is loaded in-process, nothing has to be deployed per checkpoint, which is
        what makes sweeping cheap -- with a server each checkpoint would pay a full engine startup.
        """
        args = self.args
        if not args.eval_offline:
            raise ValueError('eval_ckpts requires eval_offline; sweeping deployed servers is not supported.')

        ckpt_dirs = []
        for pattern in args.eval_ckpts:
            matched = sorted(glob.glob(pattern))
            ckpt_dirs.extend(matched or [pattern])

        scores = {}  # dataset -> {checkpoint: score}
        for ckpt_dir in ckpt_dirs:
            logger.info(f'Evaluating checkpoint: {ckpt_dir}')
            ckpt_args = copy.copy(args)
            if os.path.exists(os.path.join(ckpt_dir, 'adapter_config.json')):
                ckpt_args.adapters = [ckpt_dir]
            else:
                ckpt_args.model = ckpt_dir
                ckpt_args.adapters = []
            report = run_task(task_cfg=self.get_offline_task_cfg(ckpt_args))
            for dataset, dataset_report in report.items():
                scores.setdefault(dataset, {})[ckpt_dir] = dataset_report.score
            # release the checkpoint before loading the next one
            gc_collect()

        self.write_ckpt_report(scores)
        return scores

    def write_ckpt_report(self, scores):
        """Write a `checkpoint x dataset` table, marking the best checkpoint for each dataset."""
        args = self.args
        datasets = list(scores)
        lines = [
            '| checkpoint |' + ''.join(f' {dataset} |' for dataset in datasets),
            '|---|' + '|'.join(['---'] * len(datasets)) + '|'
        ]
        ckpt_dirs = list({ckpt_dir for per_ckpt in scores.values() for ckpt_dir in per_ckpt})
        for ckpt_dir in ckpt_dirs:
            row = [f'| `{ckpt_dir}` |']
            for dataset in datasets:
                value = scores[dataset].get(ckpt_dir)
                best = max(scores[dataset].values())
                row.append(f' {value}{" ✅" if value == best else ""} |')
            lines.append(''.join(row))

        report_path = args.eval_report or os.path.join(args.eval_output_dir, 'benchmark_report.md')
        os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f'The checkpoint sweep report has been saved to: `{report_path}`.')

    def get_task_cfg(self, dataset: List[str], eval_backend: str, url: str):
        assert eval_backend in {EvalBackend.NATIVE, EvalBackend.OPEN_COMPASS, EvalBackend.VLM_EVAL_KIT}
        if eval_backend == EvalBackend.OPEN_COMPASS:
            if self.args.local_dataset:
                if os.path.exists('data'):
                    if not os.path.exists(os.path.join('data', 'CMB')):
                        raise RuntimeError('Opencompass need a `data` folder in your work dir('
                                           'which will be created automatically by swift eval), '
                                           'but a local path named `data` already exists, '
                                           'please consider moving the dir to another location.')
                else:
                    local_dir = MediaResource.download(
                        'https://modelscope.cn/datasets/'
                        'opencompass/OpenCompassDataComplete/'
                        'resolve/master/OpenCompassData-complete-20240207.zip', 'OpenCompassData')
                    os.symlink(os.path.join(local_dir, 'data'), 'data')

            task_cfg = self.get_opencompass_task_cfg(dataset, url)
        elif eval_backend == EvalBackend.VLM_EVAL_KIT:
            task_cfg = self.get_vlmeval_task_cfg(dataset, url)
        else:
            task_cfg = self.get_native_task_cfg(dataset, url)
        return task_cfg

    def get_native_task_cfg(self, dataset: List[str], url: str):
        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'native')
        return TaskConfig(
            model=args.model_suffix,
            eval_type=EvalType.SERVICE,
            api_url=url,
            api_key=args.api_key or 'EMPTY',
            datasets=dataset,
            work_dir=work_dir,
            limit=args.eval_limit,
            eval_batch_size=args.eval_num_proc,
            dataset_args=args.eval_dataset_args,
            generation_config=args.eval_generation_config,
            **args.extra_eval_args)

    def get_opencompass_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'opencompass')
        return TaskConfig(
            eval_backend=EvalBackend.OPEN_COMPASS,
            eval_config={
                'datasets':
                dataset,
                'batch_size':
                args.eval_num_proc,
                'work_dir':
                work_dir,
                'models': [{
                    'path': args.model_suffix,
                    'openai_api_base': url,
                    'key': args.api_key or 'EMPTY',
                    'is_chat': args.use_chat_template
                }],
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)

    def get_vlmeval_task_cfg(self, dataset: List[str], url: str):
        # Must use chat/completion endpoint
        url = f"{url.rstrip('/')}/chat/completions"

        args = self.args
        work_dir = os.path.join(args.eval_output_dir, 'vlmeval')
        return TaskConfig(
            eval_backend=EvalBackend.VLM_EVAL_KIT,
            eval_config={
                'data':
                dataset,
                'model': [{
                    'type': args.model_suffix,
                    'name': 'CustomAPIModel',
                    'api_base': url,
                    'key': args.api_key or 'EMPTY',
                    **args.eval_generation_config
                }],
                'nproc':
                args.eval_num_proc,
                'limit':
                args.eval_limit
            },
            work_dir=work_dir)


def eval_main(args: Optional[Union[List[str], EvalArguments]] = None):
    return SwiftEval(args).main()
