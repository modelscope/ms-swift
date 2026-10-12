# StartLux decision LoRA training on Ascend

This example fine-tunes the language backbone of an existing StartLux decision checkpoint. It preserves the official prompt renderer and optimizes soft cross-entropy over legal answer letters. It does not append the target answer to the input, train the vision tower, or turn decisions into a text-generation SFT task.

| Item | Recipe |
| --- | --- |
| Framework | SWIFT; existing Qwen3.5 dense/MoE loading and NPU kernels |
| Checkpoints | StartLux-Decision-4B and StartLux-Decision-35B-A3B |
| Trainable weights | Rank 16 / alpha 32 / dropout 0.05; default attention LoRA, optional `--lora-scope language-linear` for all supported language-backbone linear layers |
| Frozen weights | Original backbone weights, vision, embeddings and vocabulary projection |
| Objective | Per-question soft cross-entropy at the last real prompt token; only legal candidates enter softmax |
| Numerical head | 26 letter rows projected in FP32; original frozen weight/name retained for PEFT compatibility |
| Scope | Text only, 2–26 candidates per question; unsupported wide choices/images fail explicitly |
| Output | PEFT adapter and full training-state checkpoint; not a merged standalone checkpoint |

## Environment

Use a new container. The reproduction image is pinned below; exporting the image is unnecessary. Replace all paths, container name and assigned physical NPU IDs. The driver must be installed on the host.

```bash
export IMAGE=quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1
export CONTAINER='<NEW_CONTAINER_NAME>'
export TASK_ROOT='<ABSOLUTE_TASK_DIRECTORY>'
export MODEL_ROOT='<ABSOLUTE_CHECKPOINT_PARENT>'
export DATA_ROOT='<ABSOLUTE_DATA_DIRECTORY>'
export NPU_IDS='<ASSIGNED_PHYSICAL_NPU_IDS>'
mkdir -p "$TASK_ROOT"
git clone --branch startlux_decision_npu_training \
  https://github.com/Liuchenbing-2026/ms-swift.git "$TASK_ROOT/framework"
docker pull "$IMAGE"
readarray -td, CARDS <<<"$NPU_IDS,"
DEVICES=()
for card in "${CARDS[@]}"; do
  [[ "$card" =~ ^[0-9]+$ ]] && DEVICES+=(--device "/dev/davinci$card")
done
docker run -d --name "$CONTAINER" --network host --shm-size 16g \
  "${DEVICES[@]}" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS" \
  -v "$TASK_ROOT:/workspace" -v "$MODEL_ROOT:/models:ro" -v "$DATA_ROOT:/data:ro" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$(command -v npu-smi):/usr/local/bin/npu-smi:ro" "$IMAGE" sleep infinity
docker exec -it "$CONTAINER" bash
```

The training checkout is mounted at `/workspace/framework`. Inside the container, prepare the official protocol source and Python environment:

```bash
cd /workspace
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://github.com/StartLuxLabs/StartLux-Decision.git source
git -C source checkout 0e7a2e81b9c92756e26d8edd843a44d50e362669
python -m venv --system-site-packages .venv
export PATH=/workspace/.venv/bin:$PATH
export PYTHONPATH=/workspace/framework:/workspace/source${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=4
python -m pip install --no-deps peft==0.20.0 numpy==1.26.4
python -m pip install --no-deps -e /workspace/framework
```

The tested stack uses CANN 9.1.0, PyTorch 2.10.0, torch-npu 2.10.0.post2, Transformers 5.15.1, Accelerate 1.14.0 and datasets 4.8.4. SWIFT uses PEFT 0.20.0 and NumPy 1.26.4.
The image includes NPU FLA support. Do not install CUDA FLA/causal-conv1d wheels over it. Keep CANN's Python paths when prepending the source checkout.

The original checkpoint directories must include the tokenizer and `decision_config.json`. Published references are `StartLuxAI/StartLux-Decision-4B` at `9113683823edce9311109fff84cf9719394835d8` and `StartLuxAI/StartLux-Decision-35B-A3B` at `e6dafa13615ba6858e9489416216a9b46c36e45c` on ModelScope. Download them to the host model directory before creating the read-only model mount.

## Data

Supply JSONL cases with `state`, `questions` and `targets`. State may be a text string or structured JSON business evidence; the official renderer serializes it. Image/video blocks and their reserved media keys are rejected. Candidate order is preserved by the official renderer. Each target is a distribution over the original option IDs. Values are finite, nonnegative and normalized by the loader. The optional `gold` field is mandatory for independent accuracy evaluation and contains the original hard label, which may differ from a tie-broken soft-target argmax.

```json
{"state":"A duplicate charge is visible on the receipt.","questions":{"route":{"type":"choice","instructions":"Choose the team.","criteria":{"billing":null,"shipping":null}}},"targets":{"route":{"billing":0.95,"shipping":0.05}},"gold":{"route":"billing"}}
```

Use the official `from_systemone` schema for `noul` (IDs `true`/`false`) and `score` (IDs `0`–`5`). Keep train, validation, calibration and test cases disjoint **before** flattening their questions. The scripts reject duplicate cases and train/evaluation overlap. They do not construct or relabel your dataset. Calibration/test records must not be training or checkpoint-selection inputs.

## Train and resume

Run from the example directory. Use a fresh output directory for each experiment. Start with a short run, inspect finite loss/gradients and saved weights, then run the actual domain fine-tuning experiment.

```bash
cd /workspace/framework/examples/train/startlux_decision
python train.py --model /models/StartLux-Decision-4B \
  --train /data/train.jsonl --dev /data/validation.jsonl \
  --steps 120 --batch-size 2 --accum 4 --lr 2e-5 \
  --output /workspace/results/4b-run1

# Four-device MoE run. MASTER_PORT must be free on this host.
OMP_NUM_THREADS=16 python -m torch.distributed.run --nproc-per-node 4 \
  --master-addr 127.0.0.1 --master-port 29613 train.py \
  --model /models/StartLux-Decision-35B-A3B \
  --train /data/train.jsonl --dev /data/validation.jsonl \
  --steps 120 --batch-size 1 --accum 4 --lr 2e-5 \
  --output /workspace/results/35b-run1 --fsdp
```

For SWIFT, effective global batch is batch size × accumulation × data-parallel world size; FSDP shards parameters across the four ranks. The current NPU FSDP preparation uses FP32 master weights, so two 64-GB devices are insufficient for 35B; start with four devices and adequate host RAM for CPU loading.
Resume using the **full checkpoint**, not the adapter-only export. Keep model, data, batch/accumulation and process layout unchanged; resume from an intermediate checkpoint with the original total-step and scheduler configuration:

```bash
python train.py --model /models/StartLux-Decision-4B \
  --train /data/train.jsonl --dev /data/validation.jsonl \
  --steps 120 --batch-size 2 --accum 4 --lr 2e-5 \
  --resume /workspace/results/4b-run1/checkpoint-50 \
  --output /workspace/results/4b-resumed
```

The optional `--lora-scope language-linear` follows the official fine-tuning scope for supported linear modules, including dense/shared feed-forward projections. It excludes vision and the vocabulary head. Fused MoE expert parameter tensors are not ordinary linear modules and are not selected. Keep the scope unchanged when resuming.

A short smoke run checks execution, not model quality. `validation.json` records pre/post loss, arguments and the trainable-parameter audit. Original fitted temperatures do not remain validated after an adapter changes weights.

## Independent comparison

`evaluate.py` reloads the saved adapter and compares it with the same checkpoint with the adapter disabled, using identical prompts. It requires hard labels and rejects overlap with training cases. Report absolute correct/total and the paired accuracy delta; retain regressions as well as improvements.

```bash
python evaluate.py --model /models/StartLux-Decision-4B \
  --adapter /workspace/results/4b-run1/adapter \
  --train /data/train.jsonl --data /data/test-gold.jsonl \
  --output /workspace/results/4b-test-paired

cd /workspace/framework
python -m unittest discover -s tests/trainers -p test_startlux_decision.py -v
```

Outputs contain per-question predictions, checkpoint/data hashes, correct/total and mean soft CE. This domain score is **not** the unweighted average of the official seven suites. Evaluate those suites separately using their official prompts, score aggregation and fixed checkpoint. Do not claim a general benchmark gain based on training loss or a small domain test. Before probability-based routing, fit a new temperature on a disjoint calibration split and verify it on held-out data.

## Validation status

Both real 4B and 35B checkpoints have completed forward/backward, finite optimizer updates and checkpoint save on Ascend. SWIFT 35B was checked on four devices; Twinkle 35B on two. SWIFT full-state resume completed for both sizes, but its distributed 35B run is not bitwise identical to uninterrupted training. Twinkle 4B full-state resume after a completed accumulation group reproduced all adapter tensors exactly; its distributed 35B restore is still under validation.

The included CPU tests cover candidate-only FP32 projection, causal readout, target masking, structured text input, train/evaluation overlap, original hard labels and LoRA exclusion of vision/output weights. These tests and short hardware runs establish execution contracts; independent accuracy gains and seven-suite acceptance remain experimental. No seven-suite improvement is claimed.
