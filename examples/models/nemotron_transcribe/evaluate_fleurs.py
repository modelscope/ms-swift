# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import os
import re
from itertools import islice

import torch
from modelscope.msdatasets import MsDataset
from safetensors import safe_open
from swift.dataset.dataset.mllm import FleursCantonesePreprocessor


def load_model(model_path, base_model=None):
    from nemo.collections.speechlm2.models import SALMAutomodel

    if os.path.isfile(os.path.join(model_path, 'model.safetensors')):
        return SALMAutomodel.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f'No safetensors checkpoint found in {model_path}.')
    if base_model is None:
        with open(os.path.join(model_path, 'args.json'), encoding='utf-8') as f:
            base_model = json.load(f)['model']

    with open(index_path, encoding='utf-8') as f:
        weight_map = json.load(f)['weight_map']
    model = SALMAutomodel.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    loaded = set()
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(model_path, shard), framework='pt', device='cpu') as f:
            keys = set(f.keys())
            model.load_state_dict({key: f.get_tensor(key) for key in keys}, strict=False)
            loaded.update(keys)
    model_keys = set(model.state_dict())
    missing = model_keys - loaded
    if missing:
        raise ValueError(f'Checkpoint is missing model weights: {sorted(missing)[:10]}.')
    extra = loaded - model_keys
    if extra:
        print(f'Ignoring {len(extra)} checkpoint tensors not owned by the inference model.')
    return model


def normalize(text):
    return re.sub(r'\s+', '', text)


def edit_distance(prediction, reference):
    previous = list(range(len(reference) + 1))
    for i, pred_char in enumerate(prediction, 1):
        current = [i]
        for j, ref_char in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (pred_char != ref_char)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Cantonese transcription CER on the held-out FLEURS test split.')
    parser.add_argument('--model', required=True, help='Base or merged NeMo SpeechLM checkpoint.')
    parser.add_argument('--base_model', help='Base checkpoint for resolving sharded merged weights.')
    parser.add_argument('--output', required=True, help='JSONL file for predictions and references.')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    dataset = MsDataset.load('google/fleurs', subset_name='yue_hant_hk', split='test', use_streaming=True)
    model = load_model(args.model, args.base_model).cuda().eval()

    total_errors = 0
    total_characters = 0
    count = 0
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as writer:
        for row in islice(dataset, args.num_samples):
            prompt = f'{model.audio_locator_tag}{FleursCantonesePreprocessor.prompt}'
            audio = row['audio'].get_all_samples()
            if audio.sample_rate != model.sampling_rate:
                raise ValueError(f'Expected {model.sampling_rate} Hz audio, got {audio.sample_rate} Hz.')
            waveform = audio.data.mean(dim=0, keepdim=True).to(model.device)
            answer_ids = model.generate(
                prompts=[[{
                    'role': 'user',
                    'content': prompt,
                }]],
                audios=waveform,
                audio_lens=torch.tensor([waveform.shape[1]], device=model.device),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                enable_thinking=False,
            )
            prediction = model.tokenizer.ids_to_text(answer_ids[0]).strip()
            reference = row['raw_transcription']
            errors = edit_distance(normalize(prediction), normalize(reference))
            total_errors += errors
            total_characters += len(normalize(reference))
            count += 1
            writer.write(
                json.dumps({
                    'id': row['id'],
                    'prediction': prediction,
                    'reference': reference,
                    'errors': errors,
                },
                           ensure_ascii=False) + '\n')
            writer.flush()
            if count == 1 or count % 10 == 0:
                print(f'[{count}] prediction: {prediction}')
                print(f'[{count}] reference:  {reference}')

    if total_characters == 0:
        raise ValueError('The evaluation split produced no reference characters.')
    cer = total_errors / total_characters
    print(f'model={args.model}')
    print(f'samples={count} errors={total_errors} characters={total_characters} CER={cer:.4%}')


if __name__ == '__main__':
    main()
