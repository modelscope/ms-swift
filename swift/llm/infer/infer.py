# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import nullcontext
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.distributed as dist
from datasets import Dataset as HfDataset

from swift.llm import InferArguments, InferRequest, SwiftPipeline, load_dataset, prepare_model_template, sample_dataset
from swift.utils import get_logger, is_master, open_jsonl_writer
from .infer_engine import AdapterRequest, PtEngine
from .protocol import RequestConfig
from .utils import InferCliState

logger = get_logger()


class SwiftInfer(SwiftPipeline):
    args_class = InferArguments
    args: args_class

    def __init__(self, args: Union[List[str], InferArguments, None] = None) -> None:
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
            logger.info(f'model: {self.infer_engine.model}')
        else:
            self.infer_engine = self.get_infer_engine(args)
            self.template = args.get_template(self.processor)
        self.random_state = np.random.RandomState(args.data_seed)

    def __getattr__(self, key: str):
        try:
            return super().__getattr__(key)
        except AttributeError:
            if 'infer_engine' in self.__dict__:
                return getattr(self.infer_engine, key)
            raise

    @staticmethod
    def get_infer_engine(args: InferArguments, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
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
        else:
            from .infer_engine import LmdeployEngine
            infer_engine_cls = LmdeployEngine
            kwargs.update(args.get_lmdeploy_engine_kwargs())

        return infer_engine_cls(**kwargs)

    def main(self):
        args = self.args
        context = open_jsonl_writer(args.result_path) if args.result_path else nullcontext()
        with context as json_writer:
            self.jsonl_writer = json_writer
            return super().main()

    def run(self) -> List[Dict[str, Any]]:
        args = self.args
        if args.eval_human:
            result = self.infer_cli()
        else:
            result = self.infer_dataset()
        if args.result_path:
            logger.info(f'The inference results have been saved to result_path: `{args.result_path}`.')
        return result

    def infer_single(self, infer_request: Union[InferRequest, Dict[str, Any]], request_config: RequestConfig) -> str:
        res_or_gen = self.infer([infer_request],
                                request_config,
                                template=self.template,
                                use_tqdm=False,
                                **self.infer_kwargs)
        if request_config and request_config.stream:
            response = ''
            for res in res_or_gen:
                delta = res[0].choices[0].delta.content
                print(delta, end='', flush=True)
                response += delta
            print()
        else:
            response = res_or_gen[0].choices[0].message.content
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
            if args.task_type == 'seq_cls' and args.num_labels == 1:
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
                data = {'response': response, **data}
            result_list.append(data)
            if self.jsonl_writer:
                self.jsonl_writer.append(data)

        return result_list

    def _prepare_val_dataset(self) -> HfDataset:
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        if len(args.val_dataset) > 0:
            _, val_dataset = load_dataset(args.val_dataset, split_dataset_ratio=1.0, **dataset_kwargs)
        else:
            _, val_dataset = load_dataset(args.dataset, split_dataset_ratio=args.split_dataset_ratio, **dataset_kwargs)
        assert val_dataset is not None
        val_dataset = sample_dataset(val_dataset, args.val_dataset_sample, self.random_state)
        return val_dataset

    def infer_dataset(self) -> List[Dict[str, Any]]:
        args = self.args
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')
        result_list = []
        if request_config and request_config.stream:
            for data in val_dataset:
                labels = InferRequest.remove_response(data['messages'])
                query = data['messages'][-1]['content']
                print(f'[QUERY] {query}')
                if labels:
                    print(f'[LABELS] {labels}')
                print('[RESPONSE] ', end='')
                response = self.infer_single(data, request_config)
                data = {'response': response, 'labels': labels, **data}
                result_list.append(data)
                if self.jsonl_writer:
                    self.jsonl_writer.append(data)
        else:
            is_dist = args.global_world_size > 1 and dist.is_initialized()
            if is_dist:
                val_dataset = val_dataset.shard(args.global_world_size, args.rank, contiguous=True)
            val_dataset = list(val_dataset)
            labels_list = []
            for data in val_dataset:
                if args.task_type == 'causal_lm':
                    labels = InferRequest.remove_response(data['messages'])
                else:
                    labels = data.pop('label', None)
                    if labels is not None:
                        labels = str(int(labels))
                labels_list.append(labels)

            resp_list = self.infer(
                val_dataset, request_config, template=self.template, use_tqdm=True, **self.infer_kwargs)
            for data, resp, labels in zip(val_dataset, resp_list, labels_list):
                response = resp.choices[0].message.content
                data = {'response': response, 'labels': labels, 'logprobs': resp.choices[0].logprobs, **data}
                result_list.append(data)
            if is_dist:
                total_result_list = [None for _ in range(args.global_world_size)] if args.rank == 0 else None
                dist.gather_object(result_list, total_result_list)
                result_list = total_result_list and list(chain.from_iterable(total_result_list))

            if is_master() and self.jsonl_writer and result_list:
                self.jsonl_writer.append(result_list)
        return result_list


def infer_main(args: Union[List[str], InferArguments, None] = None):
    return SwiftInfer(args).main()
