import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _test_model(model_id):
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model=model_id, to_mcore=True, exist_ok=True, test_convert_precision=True))


def test_llama2():
    _test_model('modelscope/Llama-2-7b-chat-ms')


def test_llama3():
    _test_model('LLM-Research/Meta-Llama-3-8B-Instruct')


def test_marco_o1():
    _test_model('AIDC-AI/Marco-o1')


def test_deepseek_r1_llama():
    # TODO: FIX rope
    _test_model('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')


def test_deepseek_r1_qwen():
    _test_model('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')


def test_yi():
    _test_model('01ai/Yi-1.5-6B-Chat')


def test_megrez():
    _test_model('InfiniAI/Megrez-3b-Instruct')


if __name__ == '__main__':
    # test_llama2()
    # test_llama3()
    # test_marco_o1()
    # test_deepseek_r1_llama()
    # test_deepseek_r1_qwen()
    # test_yi()
    test_megrez()
