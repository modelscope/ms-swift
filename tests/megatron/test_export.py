import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _infer_model(pt_engine, system=None, messages=None):
    from swift.utils import seed_everything, get_logger
    from swift.llm import RequestConfig
    logger = get_logger()
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    else:
        messages = messages.copy()
    resp = pt_engine.infer([{
        'messages': messages,
    }], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


model_id = 'Qwen/Qwen2-7B-Instruct'


def hf2megatron():
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model=model_id, to_megatron=True, torch_dtype='bfloat16'))


def megatron2hf():
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(megatron_model='Qwen2-7B-Instruct-megatron', to_hf=True, torch_dtype='bfloat16'))


def infer_hf_align():
    from swift.llm import PtEngine
    pt_engine = PtEngine(model_id)
    response = _infer_model(pt_engine)
    pt_engine = PtEngine('Qwen2-7B-Instruct-megatron-hf')
    response2 = _infer_model(pt_engine)
    assert response == response2

def test_train():
    from swift.llm import sft_main, TrainArguments
    sft_main(TrainArguments(megatron_model='Qwen2-7B-Instruct-megatron', dataset='AI-ModelScope/alpaca-gpt4-data-zh',
    load_args=True))


if __name__ == '__main__':
    # hf2megatron()
    # megatron2hf()
    # infer_hf_align()
    test_train()
