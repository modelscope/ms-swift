> 联合字段训练复现使用 [JOINT_REPRODUCTION.md](JOINT_REPRODUCTION.md) 和 `run_joint.sh`。下文保留历史单字段实验说明，不能混用其训练预算或评测口径。

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


## Independent evaluation scripts

`evaluate_decision.py` is the fixed single-question evaluator used for the trained 4B checkpoints. It uses the common MS-SWIFT HF/NPU backend, including when loading a checkpoint trained with Twinkle. Run it in the SWIFT environment; no Twinkle import is needed. It requires the prepared case-level JSONL (`prepared/test.jsonl`), not the compiled training-message JSONL. Use a fresh output path:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python evaluate_decision.py \
  --checkpoint /path/to/final-hf-checkpoint \
  --data /workspace/prepared/test.jsonl --batch-size 4 \
  --output /path/to/new-results/test.json
```

`evaluate_laya_suite.py` reuses the audited typed-decision predictions and evaluates the other compatible suites with batch size 8. Supply your local frozen Laya suite bundle; the bundle, model weights and measured results are not distributed here:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python evaluate_laya_suite.py \
  --checkpoint /path/to/final-hf-checkpoint \
  --dataset /path/to/laya_suites_multi.json \
  --typed-data /workspace/prepared/test.jsonl \
  --typed-result /path/to/new-results/test.json --batch-size 8 \
  --output /path/to/new-results/laya49
```

These are the previously used 4B evaluation implementations, now included with the training recipe. They are not a new performance run, an official seven-suite joint-input evaluator, or a validated 35B evaluator. Accuracy, predictions and timing outputs must remain outside the repository. This publication was syntax-checked; it did not trigger NPU reruns.
