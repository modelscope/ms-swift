from dataclasses import dataclass

from swift.llm import get_default_template_type, get_template, get_vllm_engine, inference_vllm
from swift.utils import get_main


@dataclass
class VLLMTestArgs:
    model_type: str


def test_vllm(args: VLLMTestArgs) -> None:
    model_type = args.model_type
    llm_engine = get_vllm_engine(model_type)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)

    llm_engine.generation_config.max_new_tokens = 256

    request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
    resp_list = inference_vllm(llm_engine, template, request_list)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")


test_vllm_main = get_main(VLLMTestArgs, test_vllm)

if __name__ == '__main__':
    test_vllm_main()
