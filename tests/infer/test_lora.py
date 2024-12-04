def test_llm_pt():
    from swift.llm import PtEngine, RequestConfig, LoRARequest
    engine = PtEngine('qwen/Qwen2-7B-Instruct', max_batch_size=16)

    response_list = engine.infer([{'messages': [{'role': 'user', 'content': '你是谁'}]}], request_config=RequestConfig())
    print(response_list[0].choices[0].message)
    response_list = engine.infer(
        [{
            'messages': [{
                'role': 'user',
                'content': '你是谁'
            }]
        }],
        request_config=RequestConfig(),
        lora_request=LoRARequest(
            lora_name='lora',
            lora_path='/mnt/nas2/huangjintao.hjt/work/modelscope_swift/tutorial/output/Qwen2-7B-Instruct/checkpoint-93'
        ))
    print(response_list[0].choices[0].message)


def test_llm_vllm():
    from swift.llm import VllmEngine, RequestConfig, LoRARequest
    engine = VllmEngine('qwen/Qwen2-7B-Instruct', enable_lora=True)

    response_list = engine.infer([{'messages': [{'role': 'user', 'content': '你是谁'}]}], request_config=RequestConfig())
    print(response_list[0].choices[0].message)
    response_list = engine.infer(
        [{
            'messages': [{
                'role': 'user',
                'content': '你是谁'
            }]
        }],
        request_config=RequestConfig(),
        lora_request=LoRARequest(
            lora_name='lora',
            lora_path='/mnt/nas2/huangjintao.hjt/work/modelscope_swift/tutorial/output/Qwen2-7B-Instruct/checkpoint-93'
        ))
    print(response_list[0].choices[0].message)


if __name__ == '__main__':
    # test_llm_pt()
    test_llm_vllm()
