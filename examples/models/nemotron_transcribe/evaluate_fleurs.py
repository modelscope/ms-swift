# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import os
import re
import tempfile
import torch
from itertools import islice
from modelscope.msdatasets import MsDataset
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor, PretrainedConfig

from swift.dataset.dataset.mllm import FleursCantonesePreprocessor


def _make_checkpoint_view(model_path, base_model):
    config_dict = PretrainedConfig.get_config_dict(model_path)[0]
    if base_model is None and 'audio_config' in config_dict:
        return None
    if base_model is None:
        raise ValueError('A Megatron-merged checkpoint needs --base_model pointing to the transformed checkpoint '
                         'metadata (config/processor/tokenizer files).')
    view = tempfile.TemporaryDirectory()
    config = AutoConfig.from_pretrained(base_model)
    processor = AutoProcessor.from_pretrained(base_model)
    config.save_pretrained(view.name)
    processor.save_pretrained(view.name)
    generation_config = os.path.join(base_model, 'generation_config.json')
    if os.path.isfile(generation_config):
        os.symlink(os.path.abspath(generation_config), os.path.join(view.name, 'generation_config.json'))
    for name in os.listdir(model_path):
        if name.endswith('.safetensors') or name == 'model.safetensors.index.json':
            os.symlink(os.path.abspath(os.path.join(model_path, name)), os.path.join(view.name, name))
    return view


def load_model(model_path, base_model=None):
    adapter_config = os.path.join(model_path, 'adapter_config.json')
    if os.path.isfile(adapter_config):
        import safetensors.torch
        from peft import PeftModel

        with open(adapter_config, encoding='utf-8') as f:
            base_model = base_model or json.load(f)['base_model_name_or_path']
        processor = AutoProcessor.from_pretrained(base_model)
        model = AutoModelForMultimodalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map='auto')
        model = PeftModel.from_pretrained(model, model_path)
        vit_path = os.path.join(model_path, 'vit.safetensors')
        if os.path.isfile(vit_path):
            model.load_state_dict(safetensors.torch.load_file(vit_path), strict=False)
        model = model.merge_and_unload()
    else:
        view = _make_checkpoint_view(model_path, base_model)
        load_dir = view.name if view is not None else model_path
        processor = AutoProcessor.from_pretrained(load_dir)
        model = AutoModelForMultimodalLM.from_pretrained(load_dir, dtype=torch.bfloat16, device_map='auto')
        # Keep the temporary overlay alive for the lifetime of the loaded model.
        model._checkpoint_view = view
    model.eval()
    return model, processor


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
    parser.add_argument('--model', required=True, help='Transformed checkpoint or Megatron-merged checkpoint.')
    parser.add_argument('--base_model', help='Transformed checkpoint metadata used by a Megatron-merged checkpoint.')
    parser.add_argument('--output', required=True, help='JSONL file for predictions and references.')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    dataset = MsDataset.load('google/fleurs', subset_name='yue_hant_hk', split='test', use_streaming=True)
    model, processor = load_model(args.model, args.base_model)
    sampling_rate = processor.feature_extractor.sampling_rate

    total_errors = 0
    total_characters = 0
    count = 0
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as writer:
        for row in islice(dataset, args.num_samples):
            audio = row['audio'].get_all_samples()
            if audio.sample_rate != sampling_rate:
                raise ValueError(f'Expected {sampling_rate} Hz audio, got {audio.sample_rate} Hz.')
            waveform = audio.data.mean(dim=0).numpy()
            # `device_map='auto'` may shard the model; inputs go to its first device.
            inputs = processor.apply_transcription_request(
                waveform,
                prompt=FleursCantonesePreprocessor.prompt,
                processor_kwargs={
                    'audio_kwargs': {
                        'sampling_rate': sampling_rate
                    }
                }).to(next(model.parameters()).device)
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            prediction = processor.decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
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
