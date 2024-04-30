# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import time
from typing import List

import json
from llmuses.models.custom import CustomModel
from modelscope import GenerationConfig

from swift.utils import get_logger, get_main, seed_everything
from . import EvalArguments, inference, merge_lora, prepare_model_template

logger = get_logger()


class EvalModel(CustomModel):

    def __init__(self, args: EvalArguments, model_name, config={}, **kwargs):
        if args.eval_url is not None:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=args.eval_token,
                base_url=args.eval_url,
            )
        else:
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
        super(EvalModel, self).__init__(config={'model_id': model_name, **config}, **kwargs)
        self.model_name = model_name
        self.generation_info = {'time': 0, 'tokens': 0}

    def call_openai_chat(self, query: str, history: List, **infer_args):
        infer_args.pop('best_of', None)
        history = history or []
        messages = history
        messages.append({'role': 'user', 'content': query})
        resp = self.client.chat.completions.create(model=self.args.model_type, messages=messages, **infer_args)
        response = resp.choices[0].message.content
        return response

    def call_openai_base(self, query: str, **infer_args):
        resp = self.client.completions.create(model=self.args.model_type, prompt=query, **infer_args)
        response = resp.choices[0].message.content
        return response

    def predict(self, prompt: str, **kwargs):
        if self.args.eval_url is not None:
            assert self.args.eval_is_chat_model is not None
            infer_cfg = kwargs['infer_cfg']
            infer_cfg.pop('max_length', None)
            if 'max_new_tokens' in infer_cfg:
                infer_cfg['max_tokens'] = infer_cfg.pop('max_new_tokens')
            if 'do_sample' in infer_cfg:
                infer_cfg['temperature'] = infer_cfg['temperature'] if infer_cfg['do_sample'] else 0.
                infer_cfg.pop('do_sample', None)
            if 'repetition_penalty' in infer_cfg:
                infer_cfg['presence_penalty'] = infer_cfg.pop('repetition_penalty')
            if infer_cfg.get('limit') is not None:
                infer_cfg['n'] = infer_cfg.pop('limit')
            infer_cfg.pop('limit', None)
            if 'top_k' in infer_cfg:
                infer_cfg['best_of'] = infer_cfg.pop('top_k')
            infer_cfg.pop('top_k', None)
            infer_cfg.pop('num_beams', None)
            if self.args.eval_is_chat_model:
                system = kwargs.get('system')
                history = kwargs.get('history') or []
                if system:
                    history.insert(0, {'role': 'system', 'content': 'system'})
                response = self.call_openai_chat(prompt, history, **infer_cfg)
            else:
                response = self.call_openai_base(prompt, **infer_cfg)
        elif self.args.infer_backend == 'vllm':
            from . import inference_vllm
            request_list = [{'query': prompt, 'history': kwargs.get('history'), 'system': kwargs.get('system')}]
            if 'temperature' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.temperature = kwargs['infer_cfg']['temperature']
            if 'max_new_tokens' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.max_new_tokens = kwargs['infer_cfg']['max_new_tokens']
            if 'top_k' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_k = kwargs['infer_cfg']['top_k']
            if 'top_p' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.top_p = kwargs['infer_cfg']['top_p']
            if 'repetition_penalty' in kwargs['infer_cfg']:
                self.llm_engine.generation_config.repetition_penalty = kwargs['infer_cfg']['repetition_penalty']
            resp_list = inference_vllm(self.llm_engine, self.template, request_list)
            response = resp_list[0]['response']
            new_history = resp_list[0]['history']
        else:
            generation_info = {}
            ts = time.time()
            response, new_history = inference(
                self.model,
                self.template,
                prompt,
                history=kwargs.get('history'),
                system=kwargs.get('system'),
                generation_info=generation_info,
                generation_config=GenerationConfig(**kwargs['infer_cfg']))
            self.generation_info['time'] += time.time() - ts
            self.generation_info['tokens'] += generation_info['num_generated_tokens']

        res_d: dict = {
            'choices': [{
                'index': 0,
                'message': {
                    'content': response,
                    'role': 'assistant'
                }
            }],
            'created': int(time.time()),
            'model': self.model_name,
            'object': 'chat.completion',
        }

        return res_d


def run_eval_single_model(args: EvalArguments, model_name, record=None):
    from llmuses.run import run_task
    from llmuses.config import TaskConfig
    from llmuses.summarizer import Summarizer
    if args.eval_dataset == 'no':
        args.eval_dataset = []

    custom_names = []
    if args.custom_eval_config:
        assert os.path.isfile(args.custom_eval_config)
        with open(args.custom_eval_config, 'r') as f:
            custom_eval = json.load(f)
            for _ds in custom_eval:
                custom_names.append(_ds['name'])
                TaskConfig.registry(_ds['name'], _ds['pattern'], _ds['dataset'], subset_list=_ds.get('subset_list'))
    eval_model = EvalModel(args, model_name, config=record or {})

    task_configs = TaskConfig.load(custom_model=eval_model, tasks=args.eval_dataset + custom_names)
    for task_config in task_configs:
        task_config.use_cache = args.eval_use_cache
        if args.eval_limit:
            task_config.limit = args.eval_limit
        if args.eval_few_shot is not None:
            for dataset in task_config.datasets:
                if not task_config.dataset_args.get(dataset):
                    task_config.dataset_args[dataset] = {}
                task_config.dataset_args[dataset]['few_shot_num'] = args.eval_few_shot
    logger.warn('Eval does not support temperature/top_p/do_sample argument')
    logger.info(f'Eval task config: {task_configs}')
    run_task(task_cfg=task_configs)
    final_report: List[dict] = Summarizer.get_report_from_cfg(task_cfg=task_configs)
    final_report = {
        'report': final_report,
        'generation_info': eval_model.generation_info,
    }
    print(f'Final report:{final_report}\n', flush=True)
    return final_report


def llm_eval(args: EvalArguments) -> None:
    logger.info(f'args: {args}')
    seed_everything(args.seed)
    model_name = args.model_type
    if args.name:
        model_name += f'-{args.name}'
    run_eval_single_model(args, model_name)


eval_main = get_main(EvalArguments, llm_eval)
