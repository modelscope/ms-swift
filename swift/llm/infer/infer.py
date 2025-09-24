# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from tqdm import tqdm

from swift.llm import InferArguments, InferRequest, SwiftPipeline, load_dataset, prepare_model_template, sample_dataset
from swift.plugin import InferStats, MeanMetric, compute_rouge_bleu
from swift.utils import JsonlWriter, get_dist_setting, get_logger, is_dist, is_master, read_from_jsonl
from .infer_engine import AdapterRequest, PtEngine
from .protocol import RequestConfig
from .utils import InferCliState

logger = get_logger()


class SwiftInfer(SwiftPipeline):
    args_class = InferArguments
    args: args_class

    def __init__(self, args: Optional[Union[List[str], InferArguments]] = None) -> None:
        from swift.llm import merge_lora
        super().__init__(args)
        args = self.args
        if args.merge_lora:
            merge_lora(args, device_map='cpu')
        self.infer_kwargs = {}
        if args.infer_backend == 'vllm' and args.adapters:
            self.infer_kwargs['adapter_request'] = AdapterRequest('_lora', args.adapters[0])

        if args.infer_backend == 'pt':
            model, self.template = prepare_model_template(args)
            self.infer_engine = PtEngine.from_model_template(model, self.template, max_batch_size=args.max_batch_size)
            self.infer_engine.reranker_use_activation = args.reranker_use_activation
            logger.info(f'model: {self.infer_engine.model}')
        else:
            self.template = args.get_template(None)
            self.infer_engine = self.get_infer_engine(args, self.template)
        self.random_state = np.random.RandomState(args.data_seed)

    def __getattr__(self, key: str):
        try:
            return super().__getattr__(key)
        except AttributeError:
            if 'infer_engine' in self.__dict__:
                return getattr(self.infer_engine, key)
            raise

    @staticmethod
    def get_infer_engine(args: InferArguments, template=None, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'template': template,
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        if infer_backend in {'pt', 'vllm'}:
            kwargs['reranker_use_activation'] = args.reranker_use_activation
        if infer_backend == 'pt':
            from .infer_engine import PtEngine
            infer_engine_cls = PtEngine
            kwargs.update(args.get_model_kwargs())
            if hasattr(args, 'max_batch_size'):
                kwargs.update({'max_batch_size': args.max_batch_size})
        elif infer_backend == 'vllm':
            from .infer_engine import VllmEngine
            infer_engine_cls = VllmEngine
            kwargs.update(args.get_vllm_engine_kwargs())
            seed = args.seed
            if is_dist():
                # Ensure that different data-parallel processes have different seeds.
                seed += get_dist_setting()[0] // args.vllm_tensor_parallel_size
                kwargs['distributed_executor_backend'] = 'external_launcher'
            kwargs['seed'] = seed
        elif infer_backend == 'sglang':
            from .infer_engine import SglangEngine
            infer_engine_cls = SglangEngine
            kwargs.update(args.get_sglang_engine_kwargs())
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update(args.get_lmdeploy_engine_kwargs())
        return infer_engine_cls(**kwargs)

    def run(self) -> List[Dict[str, Any]]:
        args = self.args
        self.jsonl_writer = JsonlWriter(args.result_path) if args.result_path else None
        if args.eval_human:
            result = self.infer_cli()
        else:
            result = self.infer_dataset()
        if args.result_path:
            logger.info(f'The inference results have been saved to result_path: `{args.result_path}`.')
        return result

    @staticmethod
    def parse_data_from_response(response):
        if hasattr(response, 'choices'):
            return response.choices[0].message.content
        elif hasattr(response, 'data'):
            emb = response.data[0].embedding
            shape = len(emb)
            sample = str(emb)
            if len(emb) > 6:
                sample = str(emb[:3])[:-1] + ', ..., ' + str(emb[-3:])[1:]
            return f'Embedding(shape: [1, {shape}]): {sample}'

    def infer_single(self, infer_request: Union[InferRequest, Dict[str, Any]], request_config: RequestConfig) -> str:
        res_or_gen = self.infer([infer_request],
                                request_config,
                                template=self.template,
                                use_tqdm=False,
                                **self.infer_kwargs)[0]
        if request_config and request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res.choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
            print()
        else:
            response = self.parse_data_from_response(res_or_gen)
            print(response)
        print('-' * 50)
        return response

    def infer_cli(self) -> List[Dict[str, Any]]:
        args = self.args
        template = self.template
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        logger.info('Input `exit` or `quit` to exit the conversation.')
        logger.info('Input `multi-line` to switch to multi-line input mode.')
        logger.info('Input `reset-system` to reset the system and clear the history.')
        support_multi_round = template.template_meta.support_multi_round
        if support_multi_round:
            logger.info('Input `clear` to clear the history.')
        else:
            logger.info('The current template only supports single-round dialogues.')

        infer_state = InferCliState()
        result_list = []
        while True:
            if not support_multi_round:
                infer_state.clear()
            query = infer_state.input_text()
            if query.strip().lower() in {'exit', 'quit'}:
                break
            query = infer_state.check_query(query)
            if query is None:
                continue
            infer_state.add_query(query)
            if args.model_meta.is_multimodal:
                infer_state.input_mm_data()
            if args.model_meta.is_reward or args.task_type == 'prm':
                # reward model
                response = infer_state.input_text()
                infer_state.add_response(response)
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                data = {'response': response, **data}
            else:
                data = infer_state.to_dict()
                response = self.infer_single(data, request_config)
                infer_state.add_response(response)
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, **data}
            result_list.append(data)
            if self.jsonl_writer:
                self.jsonl_writer.append(data)

        return result_list

    def _prepare_val_dataset(self) -> HfDataset:
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
        else:
            _, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        assert val_dataset is not None
        val_dataset = sample_dataset(val_dataset, args.val_dataset_sample, args.dataset_shuffle, self.random_state)
        return val_dataset

    def _calc_metric(self):
        args = self.args
        if not is_master():
            return
        data_list = read_from_jsonl(self.jsonl_writer.fpath)
        preds, labels = [], []
        for data in data_list:
            preds.append(data['response'])
            labels.append(data['labels'])
        if args.metric == 'acc':
            mean_metric = MeanMetric()
            for pred, label in zip(preds, labels):
                mean_metric.update(pred == label)
            res = {'acc': mean_metric.compute()['value']}
        elif args.metric == 'rouge':
            res = compute_rouge_bleu(preds, labels)
        logger.info(res)

    def infer_dataset(self) -> List[Dict[str, Any]]:
        args = self.args
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')

        self.infer_kwargs['metrics'] = [InferStats()]
        if request_config and request_config.stream:
            result_list = []
            for data in val_dataset:
                labels = InferRequest.remove_response(data['messages'])
                query = data['messages'][-1]['content']
                print(f'[QUERY] {query}')
                if labels:
                    print(f'[LABELS] {labels}')
                print('[RESPONSE] ', end='')
                response = self.infer_single(data, request_config)
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, 'labels': labels, **data}
                result_list.append(data)
                if self.jsonl_writer:
                    self.jsonl_writer.append(data)
            metrics = self.infer_kwargs.pop('metrics')
            print(metrics[0].compute())
        else:
            if args.write_batch_size <= 0:
                args.write_batch_size = len(val_dataset)
            if args.write_batch_size < len(val_dataset) and args.result_path:
                logger.info(f'args.result_path: {args.result_path}')
            prog_bar = tqdm(
                total=len(val_dataset), dynamic_ncols=True, disable=args.write_batch_size >= len(val_dataset))
            result_list = []
            idx = 0
            while idx < len(val_dataset):
                shard_size = min(args.write_batch_size, len(val_dataset) - idx)
                shard_dataset = val_dataset.select(range(idx, idx + shard_size))
                result_list += self._batch_infer(shard_dataset, request_config)
                idx += shard_size
                prog_bar.update(shard_size)
            prog_bar.close()
            metrics = self.infer_kwargs.pop('metrics')
            if result_list:
                metric = metrics[0].compute()
                print(f'[rank{args.rank}] {metric}' if args.rank >= 0 else str(metric))
        if args.metric is not None:
            self._calc_metric()
        return result_list

    def _batch_infer(self, val_dataset, request_config):
        args = self.args
        result_list = []
        if args.infer_backend == 'vllm':
            rank = args.rank // args.vllm_tensor_parallel_size if args.rank >= 0 else -1
            data_parallel_size = args.global_world_size // args.vllm_tensor_parallel_size
        else:
            rank, data_parallel_size = args.rank, args.global_world_size
        if rank >= 0 and data_parallel_size > 1:
            val_dataset = val_dataset.shard(data_parallel_size, rank, contiguous=True)
        val_dataset = list(val_dataset)
        labels_list = []
        for data in val_dataset:
            if args.task_type == 'causal_lm':
                labels = InferRequest.remove_response(data['messages'])
            else:
                labels = data.pop('label', None)
            labels_list.append(labels)

        resp_list = self.infer(val_dataset, request_config, template=self.template, use_tqdm=True, **self.infer_kwargs)
        if not (args.infer_backend == 'vllm' and rank >= 0 and args.rank % args.vllm_tensor_parallel_size != 0):
            for data, resp, labels in zip(val_dataset, resp_list, labels_list):
                response = resp.choices[0].message.content
                data['messages'].append({'role': 'assistant', 'content': response})
                data = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data}
                result_list.append(data)
        if self.jsonl_writer:
            self.jsonl_writer.append(result_list, gather_obj=True)
        return result_list


def infer_main(args: Optional[Union[List[str], InferArguments]] = None):
    return SwiftInfer(args).main()
