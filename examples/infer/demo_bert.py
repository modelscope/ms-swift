# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    resp_list = engine.infer(infer_requests)
    query0 = infer_requests[0].messages[0]['content']
    query1 = infer_requests[1].messages[0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'query1: {query1}')
    print(f'response1: {resp_list[1].choices[0].message.content}')


if __name__ == '__main__':
    # This is an example of BERT with LoRA.
    from swift.llm import InferEngine, InferRequest, PtEngine, load_dataset, safe_snapshot_download, BaseArguments
    from swift.tuners import Swift
    adapter_path = safe_snapshot_download('swift/test_bert')
    args = BaseArguments.from_pretrained(adapter_path)
    args.max_length = 512
    args.truncation_strategy = 'right'
    # method1
    model, processor = args.get_model_processor()
    model = Swift.from_pretrained(model, adapter_path)
    template = args.get_template(processor)
    engine = PtEngine.from_model_template(model, template, max_batch_size=64)

    # method2
    # engine = PtEngine(args.model, adapters=[adapter_path], max_batch_size=64,
    #                   task_type=args.task_type, num_labels=args.num_labels)
    # template = args.get_template(engine.processor)
    # engine.default_template = template

    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset(['DAMO_NLP/jd:cls#1000'], seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(messages=data['messages']) for data in dataset]
    infer_batch(engine, infer_requests)

    infer_batch(engine, [
        InferRequest(messages=[{
            'role': 'user',
            'content': '今天天气真好呀'
        }]),
        InferRequest(messages=[{
            'role': 'user',
            'content': '真倒霉'
        }])
    ])
