# Joint decision training with MS-SWIFT

## Objective and scope

Start from Qwen3.5-4B instruction weights, train all language parameters and freeze vision/projector parameters. Input messages retain the entire joint JSON skeleton. Gold answer symbols are supplied separately as `decision_targets`, one single-token symbol per marker, in field order. The logit immediately before a marker predicts its answer. The existing `qwen3_5` model loader is reused. `decision_plugin.py` registers an opt-in template through `--external_plugins`; HF causal loss shifts the labels. No new model architecture or core trainer change is required.

Images, 35B models, sequence parallelism and cross-example packing are outside the validated recipe. Preserve question and option order; do not truncate evidence.

## Environment

Historical four-NPU validation used Ascend 910B, Python 3.12, CANN 9.1.0, Torch 2.10.0+cpu, torch_npu 2.10.0.post2, Transformers 5.15.1 and Accelerate 1.14.0. Twinkle additionally used PEFT 0.19.0. Use the repository installation instructions for the selected platform.

The historical base image is `quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`. The optional `create_container.sh` reproduces its original fixed four-device layout (devices 0–3); it is not a portable scheduler or a complete dependency lock. Inspect device assignment before using that helper. No image export is required.

## Prepare inputs

From this example directory, supply already prepared case-level splits and the reference Intern-Decision schema:

```bash
python prepare_joint.py --prepared /path/to/case-splits \
  --output /path/to/new-joint-data \
  --official-schema /path/to/Intern-Decision/src/inputs/schema.py
```

The preparation tool checks compiled messages and targets against the supplied reference, records its SHA256, rejects duplicate/cross-split inputs and refuses an existing output directory. The historical schema hash was `d2fce8ffb8af19dca08a04eb01f4daae70ed43311d73c052ddcab797346d79bd`.

The historical split contained 960 train, 120 validation, 120 calibration and 400 test cases, with five fields per case. Test and calibration records never enter training or checkpoint selection. The unpublished original training corpus is not reconstructed.

## Launch

The following variables override machine-specific paths and device assignment:

```bash
export PYTHON_BIN=/path/to/training-venv/bin/python
export MODEL_PATH=/path/to/Qwen3.5-4B
export DATA_ROOT=/path/to/new-joint-data
export OUTPUT_ROOT=/path/to/new-training-run
export ASCEND_RT_VISIBLE_DEVICES='<FOUR_ASSIGNED_NPU_IDS>'
# Run sequentially, from this directory:
bash run_joint.sh smoke
bash run_joint.sh resume
bash run_joint.sh train
```

The fixed four-rank recipe uses global batch 16, AdamW, learning rate 2e-6, cosine minimum 2e-7, warmup 0.03, weight decay 0.01 and seed 42. With the historical 960-case data, two epochs give 120 optimizer updates. Do not reuse that step count as an epoch claim for a different dataset.

Smoke runs two steps and saves full state; resume verifies a third step in a fresh directory. Formal training saves a final model-only checkpoint. Model-only export does not support full optimizer/scheduler/RNG recovery. CPU checkpoint fences and Twinkle DCP hooks are scoped to this opt-in recipe; they are not general fixes for every backend.

## Independent evaluation

Use a separate MS-SWIFT HF/NPU inference environment to load either framework's exported weights:

```bash
ASCEND_RT_VISIBLE_DEVICES='<ONE_ASSIGNED_NPU_ID>' /path/to/eval-venv/bin/python evaluate_joint.py \
  --checkpoint /path/to/final-hf-checkpoint \
  --data /path/to/case-splits/test.jsonl --output /path/to/new-evaluation
```

Only validation loss may influence checkpoint selection. Official seven-suite evaluation is a separate protocol, not this held-out business subset. Historical full training and evaluation completed on the reproduction branch, but aggregate quality remained below the official target. Current PR-head device validation and upstream CI are pending. Keep measured metrics and raw predictions local.
