from typing import Any, List

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelType, TemplateType
from swift.utils import is_megatron_available


def get_url_suffix(model_id):
    if ':' in model_id:
        return model_id.split(':')[0]
    return model_id


def get_cache_mapping(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
    idx = text.find('| Model ID |')
    text = text[idx:]
    text_list = text.split('\n')[2:]
    cache_mapping = {}
    for text in text_list:
        if not text:
            continue
        items = text.split('|')
        if len(items) < 6:
            continue
        cache_mapping[items[1]] = items[5]
    return cache_mapping


def get_model_info_table():
    fpaths = ['docs/source/Instruction/支持的模型和数据集.md', 'docs/source_en/Instruction/Supported-models-and-datasets.md']
    cache_mapping = get_cache_mapping(fpaths[0])
    end_words = [['### 多模态大模型', '## 数据集'], ['### Multimodal large models', '## Datasets']]
    result = [
        '| Model ID | Model Type | Default Template | '
        'Requires | Support Megatron | Tags | HF Model ID |\n'
        '| -------- | -----------| ---------------- | '
        '-------- | ---------------- | ---- | ----------- |\n'
    ] * 2
    res_llm: List[Any] = []
    res_mllm: List[Any] = []
    mg_count_llm = 0
    mg_count_mllm = 0
    for template in TemplateType.get_template_name_list():
        assert template in TEMPLATE_MAPPING

    for model_type in ModelType.get_model_name_list():
        model_meta = MODEL_MAPPING[model_type]
        template = model_meta.template
        for group in model_meta.model_groups:
            for model in group.models:
                ms_model_id = model.ms_model_id
                hf_model_id = model.hf_model_id
                if ms_model_id:
                    ms_model_id = f'[{ms_model_id}](https://modelscope.cn/models/{get_url_suffix(ms_model_id)})'
                else:
                    ms_model_id = '-'
                if hf_model_id:
                    hf_model_id = f'[{hf_model_id}](https://huggingface.co/{get_url_suffix(hf_model_id)})'
                else:
                    hf_model_id = '-'
                tags = ', '.join(group.tags or model_meta.tags) or '-'
                requires = ', '.join(group.requires or model_meta.requires) or '-'
                if is_megatron_available():
                    from swift.megatron import model
                    support_megatron = getattr(model_meta, 'support_megatron', False)
                    for word in ['gptq', 'awq', 'bnb', 'aqlm', 'int4', 'int8', 'nf4', 'fp8']:
                        if word in ms_model_id.lower():
                            support_megatron = False
                            break
                    support_megatron = '&#x2714;' if support_megatron else '&#x2718;'
                else:
                    support_megatron = cache_mapping.get(ms_model_id, '&#x2718;')
                if support_megatron == '&#x2714;':
                    if model_meta.is_multimodal:
                        mg_count_mllm += 1
                    else:
                        mg_count_llm += 1
                r = f'|{ms_model_id}|{model_type}|{template}|{requires}|{support_megatron}|{tags}|{hf_model_id}|\n'
                if model_meta.is_multimodal:
                    res_mllm.append(r)
                else:
                    res_llm.append(r)
    print(f'LLM总数: {len(res_llm)}, MLLM总数: {len(res_mllm)}')
    print(f'[Megatron] LLM总数: {mg_count_llm}, MLLM总数: {mg_count_mllm}')
    text = ['', '']  # llm, mllm
    for i, res in enumerate([res_llm, res_mllm]):
        for r in res:
            text[i] += r
        result[i] += text[i]

    for i, fpath in enumerate(fpaths):
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        llm_start_idx = text.find('| Model ID |')
        mllm_start_idx = text[llm_start_idx + 1:].find('| Model ID |') + llm_start_idx + 1
        llm_end_idx = text.find(end_words[i][0])
        mllm_end_idx = text.find(end_words[i][1])
        output = text[:llm_start_idx] + result[0] + '\n\n' + text[llm_end_idx:mllm_start_idx] + result[
            1] + '\n\n' + text[mllm_end_idx:]
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(output)


if __name__ == '__main__':
    get_model_info_table()
