from typing import List

from swift.llm import MODEL_MAPPING, ModelType


def get_model_info_table() -> List[str]:
    model_name_list = ModelType.get_model_name_list()
    result = (
        '| Model Type | Model ID | Default Lora Target Modules | Default Template |'
        ' Support Flash Attn | Support VLLM | Requires | Tags |\n'
        '| ---------  | -------- | --------------------------- | ---------------- |'
        ' ------------------ | ------------ | -------- | ---- |\n')
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
        tags = model_info.get('tags', [])
        tags_str = ', '.join(tags)
        if len(tags_str) == 0:
            tags_str = '-'
        r = [
            model_name, model_id, lora_target_modules, template,
            support_flash_attn, support_vllm, requires, tags_str
        ]
        res.append(r)
    text = ''
    for r in res:
        url = f'https://modelscope.cn/models/{r[1]}/summary'
        text += f'|{r[0]}|[{r[1]}]({url})|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}|\n'
    print(f'模型总数: {len(res)}')
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


if __name__ == '__main__':
    result = get_model_info_table()
