from typing import Any, List

from swift.llm import MODEL_MAPPING


def get_model_info_table():
    fpaths = ['docs/source/Instruction/支持的模型和数据集.md', 'docs/source_en/Instruction/Supported-models-datasets.md']
    end_words = [['### 多模态大模型', '## 数据集'], ['### MLLM', '## Datasets']]
    result = [
        '| Model ID | HF Model ID | Model Type | Architectures | Default Template(for sft) | '
        'Requires | Tags |\n'
        '| -------- | ----------- | -----------| ------------  | ------------------------- | '
        '-------- | ---- |\n'
    ] * 2
    res_llm: List[Any] = []
    res_mllm: List[Any] = []
    for model_name, model_meta in MODEL_MAPPING.items():
        model_type = model_meta.model_type
        template = model_meta.template
        requires = model_meta.requires
        r = ''
        for group in model_meta.model_groups:
            for model in group.models:
                ms_model_id = model.ms_model_id
                hf_model_id = model.hf_model_id
                if ms_model_id:
                    ms_model_id = f'[{ms_model_id}](https://modelscope.cn/models/{ms_model_id}/summary)'
                else:
                    ms_model_id = '-'
                if hf_model_id:
                    hf_model_id = f'[{hf_model_id}](https://huggingface.co/{hf_model_id})'
                else:
                    hf_model_id = '-'
                r = (f'|{ms_model_id}|'
                     f'|{hf_model_id}|'
                     f'{model_type}|{model_meta.architectures}|{template}|'
                     f'{requires or "-"}|{group.tags or "-"}|\n')
                if model_meta.is_multimodal:
                    res_mllm.append(r)
                else:
                    res_llm.append(r)
    print(f'LLM总数: {len(res_llm)}, MLLM总数: {len(res_mllm)}')
    text = ['', '']  # llm, mllm
    for i, res in enumerate([res_llm, res_mllm]):
        for r in res:
            text[i] += r
        result[i] += text[i]

    for i, fpath in enumerate(fpaths):
        with open(fpath, 'r') as f:
            text = f.read()
        llm_start_idx = text.find('| Model ID |')
        mllm_start_idx = text[llm_start_idx + 1:].find('| Model ID |') + llm_start_idx + 1
        llm_end_idx = text.find(end_words[i][0])
        mllm_end_idx = text.find(end_words[i][1])
        output = text[:llm_start_idx] + result[0] + '\n\n' + text[llm_end_idx:mllm_start_idx] + result[
            1] + '\n\n' + text[mllm_end_idx:]
        with open(fpath, 'w') as f:
            f.write(output)


if __name__ == '__main__':
    get_model_info_table()
