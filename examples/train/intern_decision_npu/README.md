# Qwen3.5-4B text decision training on Ascend

This example updates all language parameters while freezing vision and its projector. It uses hard-label, full-vocabulary CE at decision positions; this is not LoRA and not a reconstruction of an unpublished training mixture. Use one question per example and preserve option order. Gold labels and teacher probabilities never enter the prompt.

## Environment and layout

The tested stack is CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post2, Transformers 5.15.1 and Accelerate 1.14.0 on four Ascend NPUs. Twinkle uses PEFT 0.19.0 and NumPy 1.26.4 in its separate environment. Install this repository in the chosen environment and mount this example directory as `/workspace`; mount the original Qwen3.5-4B checkpoint at `/models/Qwen3.5-4B`. Data and checkpoint locations are local mounts, not bundled artifacts. The example is text-only and does not validate image training or Qwen3.6-35B-A3B.

```bash
cd /workspace
python3 download_data.py
python3 prepare_data.py
python3 prepare_inputs.py
mkdir -p results
bash run.sh smoke
bash run.sh resume
bash run.sh train
```

Training uses four workers, global batch 16, learning rate 2e-6 and a fixed 600-step cosine schedule with 18 warmup steps. Checkpoint selection uses only validation CE. CPU checks, NPU short training, full checkpoint recovery and final independent evaluation are distinct validation stages. Keep accuracy and performance artifacts outside this repository.

Data preparation is deterministic and splits by original case; related questions remain in one split. Preserve published hard labels when rounded teacher probabilities tie. The test split is not used for training or checkpoint selection. The tokenizer encodes each answer symbol as one token; the supported native answer alphabet has 62 symbols.

## External plugins

`decision_plugin.py` registers the decision template and labels only answer positions. The framework performs the causal label shift. `checkpoint_fence.py` wraps `Trainer.save_model` with a Gloo CPU fence to keep other ranks from entering HCCL barriers during rank-zero CPU checkpoint offload. It changes runtime behavior without modifying trainer package source.

The launch recipe enables the fence from the initial run. Short training and full-state recovery must pass in the selected environment before the full run. Existing outputs must not be overwritten. Framework tests and complete upstream CI remain separate from the example's NPU validation.
