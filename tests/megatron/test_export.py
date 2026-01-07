import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _infer_model(engine, system=None, messages=None):
    from swift.utils import seed_everything, get_logger
    from swift.infer_engine import RequestConfig
    logger = get_logger()
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': 'who are you?'}]
        resp = engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    else:
        messages = messages.copy()
    resp = engine.infer([{
        'messages': messages,
    }], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {engine.model_info.model_name}, messages: {messages}')
    return response


model_id = 'Qwen/Qwen2-7B-Instruct'


def hf2mcore():
    from swift import export_main, ExportArguments
    export_main(
        ExportArguments(
            model=model_id, to_mcore=True, torch_dtype='bfloat16', exist_ok=True, test_convert_precision=True))


def mcore2hf():
    from swift import export_main, ExportArguments
    export_main(
        ExportArguments(
            mcore_model='Qwen2-7B-Instruct-mcore',
            to_hf=True,
            torch_dtype='bfloat16',
            exist_ok=True,
            test_convert_precision=True))


def infer_hf_align():
    from swift.infer_engine import TransformersEngine
    engine = TransformersEngine(model_id)
    response = _infer_model(engine)
    engine = TransformersEngine('Qwen2-7B-Instruct-mcore-hf')
    response2 = _infer_model(engine)
    assert response == response2


if __name__ == '__main__':
    # hf2mcore()
    mcore2hf()
    infer_hf_align()
