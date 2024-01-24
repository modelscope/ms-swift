import re
from typing import Dict, List, Tuple

from swift.llm import MODEL_MAPPING, ModelType


def get_model_info_table2() -> List[str]:
    model_name_list = ModelType.get_model_name_list()
    result = (
        '| Model Type | Model ID | Default Lora Target Modules | Default Template |'
        ' Support Flash Attn | Support VLLM | Requires |\n'
        '| ---------  | -------- | --------------------------- | ---------------- |'
        ' ------------------ | ------------ | -------- |\n')
    res: List[str] = []
    bool_mapping = {True: '&#x2714;', False: '&#x2718;'}
    for model_name in model_name_list:
        model_info = MODEL_MAPPING[model_name]
        model_id = model_info['model_id_or_path']
        lora_target_modules = ', '.join(model_info['lora_target_modules'])
        template = model_info['template']
        support_flash_attn = model_info.get('support_flash_attn', False)
        support_flash_attn = bool_mapping[support_flash_attn]
        support_vllm = model_info.get('support_vllm', False)
        support_vllm = bool_mapping[support_vllm]
        requires = ', '.join(model_info['requires'])
        r = [
            model_name, model_id, lora_target_modules, template,
            support_flash_attn, support_vllm, requires
        ]
        res.append(r)
    text = ''
    for r in res:
        url = f'https://modelscope.cn/models/{r[1]}/summary'
        text += f'|{r[0]}|[{r[1]}]({url})|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|\n'
    result += text
    #
    fpath = 'docs/source/LLM/支持的模型和数据集.md'
    with open(fpath, 'r') as f:
        text = f.read()
    start_idx = text.find('| Model Type |')
    end_idx = text.find('## 数据集')
    output = text[:start_idx] + result + '\n\n' + text[end_idx:]
    with open(fpath, 'w') as f:
        text = f.write(output)
    return res


def get_model_info_readme_zh(data: List[str]) -> None:
    fpath = 'README_CN.md'
    with open(fpath, 'r') as f:
        text = f.read()
    start_idx = text.find('  - 多模态:')
    end_idx = text.find('- 支持的数据集:')
    text = text[start_idx:end_idx]
    match_list = re.findall(r'- (.+)( 系列)?: (.+)', text)
    model_list = []
    for match in match_list:
        model_list += match[2].split(',')
    model_list = [model.strip() for model in model_list]
    model_type_list = [d[0] for d in data]
    print(set(model_type_list) - set(model_list))
    print(set(model_list) - set(model_type_list))


def get_model_info_readme_en(data: List[str]) -> None:
    fpath = 'README.md'
    with open(fpath, 'r') as f:
        raw_text = f.read()
    start_idx = raw_text.find('  - Multi-Modal:')
    end_idx = raw_text.find('- Supported Datasets:')
    text = raw_text[start_idx:end_idx]
    match_list = re.findall(r'- (.+)( series)?: (.+)', text)
    model_list = []
    for match in match_list:
        model_list += match[2].split(',')
    model_list = [model.strip() for model in model_list]
    model_type_list = [d[0] for d in data]
    print(set(model_type_list) - set(model_list))
    print(set(model_list) - set(model_type_list))


if __name__ == '__main__':
    result = get_model_info_table2()
    result_en = get_model_info_readme_en(result)
    result_zh = get_model_info_readme_zh(result)
