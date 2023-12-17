from typing import Dict, List, Tuple

from swift.llm import MODEL_MAPPING, ModelType


def write_model_info_table2(fpath: str) -> None:
    model_name_list = ModelType.get_model_name_list()
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(
            '| Model Type | Model ID | Default Lora Target Modules | Default Template |'
            ' Support Flash Attn | Requires |\n'
            '| ---------  | -------- | --------------------------- | ---------------- |'
            ' ------------------ | -------- |\n')
    res = []
    bool_mapping = {True: '&#x2714;', False: '&#x2718;'}
    for model_name in model_name_list:
        model_info = MODEL_MAPPING[model_name]
        model_id = model_info['model_id_or_path']
        lora_target_modules = ', '.join(model_info['lora_target_modules'])
        template = model_info['template']
        support_flash_attn = model_info.get('support_flash_attn', False)
        support_flash_attn = bool_mapping[support_flash_attn]
        requires = ', '.join(model_info['requires'])
        r = [
            model_name, model_id, lora_target_modules, template,
            support_flash_attn, requires
        ]
        res.append(r)
    text = ''
    for r in res:
        url = f'https://modelscope.cn/models/{r[1]}/summary'
        text += f'|{r[0]}|[{r[1]}]({url})|{r[2]}|{r[3]}|{r[4]}|{r[5]}|\n'
    with open(fpath, 'a', encoding='utf-8') as f:
        f.write(text)
    print()


if __name__ == '__main__':
    write_model_info_table2('model_info.md')
