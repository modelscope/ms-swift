"""Audit saved model changes and full training-state files after a short run."""
import argparse
import json
import torch
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer


def weight_map(root):
    manifest = root / 'model.safetensors.index.json'
    if manifest.exists():
        return json.loads(manifest.read_text())['weight_map']
    with safe_open(root / 'model.safetensors', framework='pt', device='cpu') as handle:
        return {key: 'model.safetensors' for key in handle.keys()}


def tensor(root, mapping, key):
    with safe_open(root / mapping[key], framework='pt', device='cpu') as handle:
        return handle.get_tensor(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--base', type=Path, default=Path('/models/Qwen3.5-4B'))
    parser.add_argument('--framework', choices=['swift', 'twinkle'], required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = weight_map(args.base), weight_map(args.checkpoint)
    # HF Qwen3_5ForConditionalGeneration does not load the base MTP predictor.
    # The training checkpoint may explicitly serialize the tied output head.
    ignored_mtp = {key for key in before if key.startswith('mtp.')}
    expected = set(before) - ignored_mtp
    extra = set(after) - expected
    assert expected <= set(after), 'A runtime parameter is missing'
    assert extra <= {'lm_head.weight'}, 'Unexpected exported parameter'
    config = json.loads((args.checkpoint / 'config.json').read_text())
    assert config['architectures'] == ['Qwen3_5ForConditionalGeneration']
    if 'lm_head.weight' in extra:
        assert config.get('tie_word_embeddings') is True
        head = tensor(args.checkpoint, after, 'lm_head.weight')
        embedding = tensor(args.checkpoint, after, 'model.language_model.embed_tokens.weight')
        torch.testing.assert_close(head, embedding, rtol=0, atol=0)
        del head, embedding
    assert all((args.checkpoint / name).is_file() for name in set(after.values()))
    vision = [key for key in before if '.visual.' in key]
    language = [key for key in before if '.language_model.layers.0.' in key and key.endswith('weight')][:5]
    assert vision and language
    report = {
        'checkpoint': str(args.checkpoint),
        'frozen_tensors_verified': len(vision),
        'unloaded_base_mtp_tensors': sorted(ignored_mtp),
        'language_sample': {}
    }
    for key in vision:
        old = tensor(args.base, before, key).float()
        new = tensor(args.checkpoint, after, key).float()
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    for key in language:
        old = tensor(args.base, before, key).float()
        new = tensor(args.checkpoint, after, key).float()
        report['language_sample'][key] = {
            'max_abs_delta': float((old - new).abs().max()),
            'changed_elements': int((old != new).sum())
        }
    assert any(row['changed_elements'] for row in report['language_sample'].values())
    marker = AutoTokenizer.from_pretrained(args.checkpoint).encode('<decision>', add_special_tokens=False)
    assert len(marker) == 1
    report['marker_id'] = marker[0]
    state = json.loads((args.checkpoint / 'trainer_state.json').read_text())
    report['global_step'] = state['global_step' if args.framework == 'swift' else 'cur_step']
    assert report['global_step'] == args.expected_step
    assert any((args.checkpoint / name).exists() for name in ['optimizer.pt', 'optimizer.bin'])
    optimizer_dir = args.checkpoint / 'optimizer.pt'
    if optimizer_dir.is_dir():
        assert (optimizer_dir / '.metadata').is_file()
        assert len(list(optimizer_dir.glob('*.distcp'))) == 4
    assert (args.checkpoint / 'scheduler.pt').exists()
    pattern = 'rng_state*.pth' if args.framework == 'swift' else 'rng_state_rank*.pt'
    assert len(list(args.checkpoint.glob(pattern))) == 4
    report['status'] = 'passed'
    report['scope'] = 'all frozen vision tensors, sampled language updates, saved state; resume validated separately'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
