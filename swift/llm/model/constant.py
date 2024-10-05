from typing import List


class LLMModelType:
    # dense
    qwen = 'qwen'
    modelscope_agent = 'modelscope_agent'
    qwen2 = 'qwen2'
    qwen2_5 = 'qwen2_5'

    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    glm4 = 'glm4'

    llama2 = 'llama2'
    llama3 = 'llama3'
    llama3_1 = 'llama3_1'
    llama3_2 = 'llama3_2'
    reflection_llama3_1 = 'reflection_llama3_1'
    chinese_llama2 = 'chinese_llama2'
    chinese_alpaca2 = 'chinese_alpaca2'
    llama3_chinese = 'llama3_chinese'

    longwriter_glm4 = 'longwriter_glm4'
    longwriter_llama3_1 = 'longwriter_llama3_1'

    atom = 'atom'

    codefuse_qwen = 'codefuse_qwen'

    # moe
    qwen2_moe = 'qwen2_moe'


class MLLMModelType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'

    glm4v = 'glm4v'
    llama3_2_vision = 'llama3_2_vision'
    llama3_1_omni = 'llama3_1_omni'
    idefics3_llama3 = 'idefics3_llama3'

    llava1_5 = 'llava1_5'
    llava1_6_mistral = 'llava1_6_mistral'
    llava1_6_vicuna = 'llava1_6_vicuna'
    llava1_6_yi = 'llava1_6_yi'
    llava1_6_llama3_1 = 'llava1_6_llama3_1'
    llava_next = 'llava_next'


class ModelType(LLMModelType, MLLMModelType):

    @classmethod
    def get_model_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__'):
                continue
            value = cls.__dict__[k]
            if isinstance(value, str):
                res.append(value)
        return res
