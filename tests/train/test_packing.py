import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 3,
}


def test_llm():
    from swift import InferArguments, SftArguments, infer_main, sft_main
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'swift/self-cognition#1000'],
            split_dataset_ratio=0.01,
            packing=True,
            max_length=4096,
            attn_impl='flash_attn',
            logging_steps=1,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_streaming():
    from swift import InferArguments, SftArguments, infer_main, sft_main
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#10000'],
            packing=True,
            max_length=4096,
            streaming=True,
            attn_impl='flash_attn',
            max_steps=100,
            dataset_num_proc=1,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_streaming_packing_split():
    """IterablePackingDataset must flatten ``truncation_strategy='split'``'s list-of-chunks
    return from ``template.encode``. Regression test for
    https://github.com/modelscope/ms-swift/issues/8285 : streaming + packing + split crashed
    with ``TypeError: list indices must be integers or slices, not str`` because
    ``_fetch_data_out_queue`` assumed ``encode`` always returns a dict."""
    import queue
    import types

    from swift.dataset.packing import IterablePackingDataset, calculate_matched_group

    # _fetch_data_out_queue only touches self._out_queue; bypass __init__ (which spawns mp workers).
    obj = types.SimpleNamespace(_out_queue=queue.Queue())

    normal = {'input_ids': list(range(100)), 'labels': [-100] * 100}  # dict: delete/left/right
    chunk_a = {'input_ids': list(range(1024)), 'labels': [-100] * 1024}  # split chunk 1
    chunk_b = {'input_ids': list(range(904)), 'labels': [-100] * 904}  # split chunk 2

    obj._out_queue.put((0, normal))  # i=0: single dict (non-split)
    obj._out_queue.put((1, [chunk_a, chunk_b]))  # i=1: split -> list of chunks
    obj._out_queue.put((2, {}))  # i=2: failed encode (MaxLengthError) -> skipped

    res = IterablePackingDataset._fetch_data_out_queue(obj, [], 3)

    # Flattened to one (chunk, length) item per chunk; failed sample skipped; input order preserved.
    assert res == [(normal, 100), (chunk_a, 1024), (chunk_b, 904)]

    # Flattened chunks feed bin-packing without error; total tokens are conserved across bins.
    bins, _ = calculate_matched_group(res, 1024, is_finished=True)
    assert sum(sum(item[1] for item in b) for b in bins) == 100 + 1024 + 904


def test_streaming_packing_split_skips_malformed_chunks():
    """Empty chunks or chunks missing ``'input_ids'`` are skipped defensively, keeping only
    valid chunks for bin-packing (mirrors RowPreprocessor's skip-bad-data behaviour)."""
    import queue
    import types

    from swift.dataset.packing import IterablePackingDataset

    obj = types.SimpleNamespace(_out_queue=queue.Queue())
    good = {'input_ids': list(range(50)), 'labels': [-100] * 50}
    empty_chunk = {}  # empty -> skipped
    no_input_ids = {'labels': [-100]}  # missing 'input_ids' -> skipped

    obj._out_queue.put((0, [good, empty_chunk, no_input_ids]))

    res = IterablePackingDataset._fetch_data_out_queue(obj, [], 1)
    assert res == [(good, 50)]


def test_mllm_streaming():
    from swift import InferArguments, SftArguments, infer_main, sft_main
    result = sft_main(
        SftArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['AI-ModelScope/LaTeX_OCR#20000'],
            packing=True,
            max_length=8192,
            streaming=True,
            attn_impl='flash_attn',
            max_steps=100,
            dataset_num_proc=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    # test_streaming()
    test_mllm_streaming()
