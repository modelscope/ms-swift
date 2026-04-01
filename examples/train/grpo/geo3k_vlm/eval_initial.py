"""Quick evaluation of Qwen3.5-2B on Geometry3K test set."""
import base64
import io
import os
import re

os.environ['MAX_PIXELS'] = '401408'
os.environ['VLLM_USE_MODELSCOPE'] = 'False'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from math_verify import parse, verify
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig


def pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'data:image/png;base64,{b64}'


def evaluate_accuracy(completion: str, answer: str) -> float:
    content_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    content_to_parse = content_match.group(1).strip() if content_match else completion
    has_answer_tag = content_match is not None
    try:
        gold_parsed = parse(answer, extraction_mode='first_match')
        if len(gold_parsed) == 0:
            return 0.0
        if has_answer_tag:
            answer_parsed = parse(content_to_parse, extraction_mode='first_match')
        else:
            answer_parsed = parse(
                content_to_parse,
                extraction_config=[LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False, malformed_operators=False,
                        basic_latex=True, boxed=True, units=True),
                    boxed_match_priority=0, try_extract_without_anchor=False)],
                extraction_mode='first_match')
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


def main():
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    model_name = 'Qwen/Qwen3.5-2B'
    print(f'Loading model: {model_name}')

    llm = LLM(model=model_name, dtype='bfloat16', gpu_memory_utilization=0.85,
              max_model_len=4096, trust_remote_code=True, limit_mm_per_prompt={'image': 1})

    print('Loading geometry3k test set...')
    ds = load_dataset('hiyouga/geometry3k', split='test', trust_remote_code=True)
    print(f'Test set size: {len(ds)}')

    system_prompt = (
        'A conversation between User and Assistant. The user asks a question, '
        'and the Assistant solves it. The assistant first thinks about the reasoning '
        'process in the mind and then provides the user with the answer. '
        'The reasoning process and answer are enclosed within <think> </think> and '
        '<answer> </answer> tags, respectively, i.e., '
        '<think> reasoning process here </think><answer> answer here </answer>')

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    n_eval = min(100, len(ds))

    conversations = []
    answers = []
    for i in range(n_eval):
        sample = ds[i]
        problem = sample['problem']
        image = sample['images'][0]
        data_url = pil_to_data_url(image)

        content_parts = []
        parts = problem.split('<image>')
        for j, part in enumerate(parts):
            if j > 0:
                content_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': data_url}
                })
            if part.strip():
                content_parts.append({'type': 'text', 'text': part})

        conversations.append([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content_parts},
        ])
        answers.append(sample['answer'])

    print(f'Running inference on {n_eval} samples...')
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params)

    correct = 0
    for i, (output, answer) in enumerate(zip(outputs, answers)):
        completion = output.outputs[0].text
        acc = evaluate_accuracy(completion, answer)
        correct += acc
        if i < 5 or (i + 1) % 20 == 0:
            tail = completion[-200:] if len(completion) > 200 else completion
            print(f'[{i+1}/{n_eval}] GT={answer}, Acc={acc:.0f} | ...{tail}')

    acc_rate = correct / n_eval
    print(f'\n{"="*50}')
    print(f'Model: {model_name}')
    print(f'Evaluated: {n_eval} samples from geometry3k test')
    print(f'Accuracy:  {acc_rate:.4f} ({acc_rate*100:.1f}%)')
    print(f'{"="*50}')
    if acc_rate > 0.8:
        print('WARNING: Too easy for competition.')
    elif acc_rate < 0.05:
        print('WARNING: Too hard for this model.')
    else:
        print(f'GOOD: {acc_rate*100:.1f}% initial accuracy, suitable for GRPO.')


if __name__ == '__main__':
    main()
