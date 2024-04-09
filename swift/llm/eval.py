# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import time
from typing import List

import json
from llmuses.models.custom import CustomModel
from modelscope import GenerationConfig

from swift.utils import get_logger, get_main
from . import (EvalArguments, inference, inference_vllm, merge_lora,
               prepare_model_template)

logger = get_logger()


class EvalModel(CustomModel):

    def __init__(self, args: EvalArguments, model_name, config={}, **kwargs):
        if args.merge_lora:
            merge_lora(args, device_map=args.merge_device_map)
        if args.infer_backend == 'vllm':
            from .utils import prepare_vllm_engine_template
            self.llm_engine, self.template = prepare_vllm_engine_template(args)
        else:
            self.model, self.template = prepare_model_template(args)
            if args.overwrite_generation_config:
                assert args.ckpt_dir is not None, 'args.ckpt_dir is not specified.'
                self.model.generation_config.save_pretrained(args.ckpt_dir)

        self.args = args
        super(EvalModel, self).__init__(
            config={
                'model_id': model_name,
                **config
            }, **kwargs)
        self.model_name = model_name
        self.generation_info = {'time': 0, 'tokens': 0}

    def predict(self, prompt: str, **kwargs):
        if self.args.infer_backend == 'vllm':
            request_list = [{
                'query': prompt,
                'history': kwargs.get('history'),
                'system': kwargs.get('system')
            }]
            if 'temperature' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.temperature = kwargs[
                    'infer_cfg']['temperature']
            if 'max_new_tokens' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.max_new_tokens = kwargs[
                    'infer_cfg']['max_new_tokens']
            if 'top_k' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_k = kwargs['infer_cfg'][
                    'top_k']
            if 'top_p' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_p = kwargs['infer_cfg'][
                    'top_p']
            if 'repetition_penalty' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.repetition_penalty = kwargs[
                    'infer_cfg']['repetition_penalty']
            resp_list = inference_vllm(self.llm_engine, self.template,
                                       request_list)
            response = resp_list[0]['response']
            new_history = resp_list[0]['history']
        else:
            generation_info = {}
            ts = time.time()
            response, new_history = inference(
                self.model,
                self.template,
                prompt,
                generation_info=generation_info,
                generation_config=GenerationConfig(**kwargs['infer_cfg']))
            self.generation_info['time'] += time.time() - ts
            self.generation_info['tokens'] += generation_info[
                'num_generated_tokens']

        res_d: dict = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': response,
                    'role': 'assistant'
                }
            }],
            'created':
            int(time.time()),
            'model':
            self.model_name,
            'object':
            'chat.completion',
        }

        return res_d


def run_eval_single_model(args: EvalArguments, model_name, record=None):
    from llmuses.run import run_task
    from llmuses.config import TaskConfig
    from llmuses.summarizer import Summarizer

    custom_names = []
    if args.custom_eval_config:
        assert os.path.isfile(args.custom_eval_config)
        with open(args.custom_eval_config, 'r') as f:
            custom_eval = json.load(f)
            for _ds in custom_eval:
                custom_names.append(_ds['name'])
                TaskConfig.registry(
                    _ds['name'],
                    _ds['pattern'],
                    _ds['dataset'],
                    subset_list=_ds.get('subset_list'))
    eval_model = EvalModel(args, model_name, config=record or {})

    task_configs = TaskConfig.load(
        custom_model=eval_model, tasks=args.eval_dataset + custom_names)
    for task_config in task_configs:
        task_config.use_cache = False
        if args.eval_limit:
            task_config.limit = args.eval_limit
    logger.warn('Eval does not support temperature/top_p/do_sample argument')
    logger.info(f'Eval task config: {task_configs}')
    run_task(task_cfg=task_configs)
    final_report: List[dict] = Summarizer.get_report_from_cfg(
        task_cfg=task_configs)
    final_report = {
        'report': final_report,
        'generation_info': eval_model.generation_info,
    }
    print(f'Final report:{final_report}\n', flush=True)
    return final_report


def llm_eval(args: EvalArguments) -> None:
    model_name = args.model_type
    if args.name:
        model_name += f'-{args.name}'
    run_eval_single_model(args, model_name)


eval_main = get_main(EvalArguments, llm_eval)
