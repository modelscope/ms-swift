from swift.llm import InferClient, InferRequest, RequestConfig


def infer_multilora(engine: InferClient, infer_request: InferRequest):
    # Dynamic LoRA
    models = engine.models
    print(f'models: {models}')
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # use lora
    resp_list = engine.infer([infer_request], request_config, model=models[1])
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')
    # origin model
    resp_list = engine.infer([infer_request], request_config, model=models[0])
    response = resp_list[0].choices[0].message.content
    print(f'response: {response}')
    # use lora
    resp_list = engine.infer([infer_request], request_config, model=models[1])
    response = resp_list[0].choices[0].message.content
    print(f'lora-response: {response}')


if __name__ == '__main__':
    engine = InferClient(host='127.0.0.1', port=8000)
    infer_request = InferRequest(messages=[{'role': 'user', 'content': '你是谁'}])
    infer_multilora(engine, infer_request)
