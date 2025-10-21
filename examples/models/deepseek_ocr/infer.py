# pip install "transformers==4.46.3" easydict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['SWIFT_DEBUG'] = '1'

if __name__ == '__main__':
    from swift.llm import InferRequest, PtEngine, RequestConfig
    engine = PtEngine('deepseek-ai/DeepSeek-OCR')
    infer_request = InferRequest(
        messages=[{
            'role': 'user',
            # or
            'content': '<image>Free OCR.',
            # "content": '<image><|grounding|>Convert the document to markdown.',
        }],
        images=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png'])
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer([infer_request], request_config=request_config)
    response = resp_list[0].choices[0].message.content

    # use stream
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    gen_list = engine.infer([infer_request], request_config=request_config)
    for chunk in gen_list[0]:
        if chunk is None:
            continue
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()
