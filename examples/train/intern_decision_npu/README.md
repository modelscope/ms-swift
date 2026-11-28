# Intern-Decision text training on Ascend

This example trains the language backbone of Qwen3.5-4B with joint decision supervision in MS-SWIFT, while freezing the vision tower and projector. Each assistant input contains a complete JSON skeleton with `<decision>` markers. Answer symbols appear only in labels. All supervised fields use full-vocabulary causal cross-entropy in the same forward pass.

The existing `qwen3_5` model loader is reused. `decision_plugin.py` registers an opt-in template through `--external_plugins`; HF causal loss shifts the labels. No new model architecture or core trainer change is required.

## Run the joint-field recipe

Use [JOINT_REPRODUCTION.md](JOINT_REPRODUCTION.md) for environment requirements, input preparation, smoke training, checkpoint recovery and the full run. The example currently requires four Ascend NPUs, text-only inputs and no sequence parallelism or cross-example packing.

```bash
# From the repository root, after installing its dependencies:
python -m pytest tests/general/test_intern_decision.py -q
```

These synthetic CPU regressions require no checkpoint or benchmark download. They cover causal loss and gradient equivalence; real-tokenizer and distributed checkpoint audits remain separate checks.

## Validation scope

The pinned historical reproduction completed four-rank short training and full-state recovery, 120 optimizer steps, final model export/reload and independent evaluation. The final PR revision still needs device validation and upstream CI. Historical training evidence must not be attributed to a different source revision.

The original Intern-Decision training corpus is not public. This recipe uses public Typed Decisions data and reproduces the supervision method, not the official training mixture or published model quality. It is not a StartLux or 35B training recipe. Keep weights, predictions and measured results outside the source tree.

## Supporting tools

- `prepare_joint.py`: compile all fields of a case together and compare them with a supplied official schema; reject split overlap and answer leakage.
- `check_checkpoint.py`: check exported model keys, frozen vision weights, sampled language updates and checkpoint artifacts.
- `evaluate_joint.py`: evaluate a saved model on held-out joint cases using the separately installed MS-SWIFT HF/NPU backend.
- `run.sh`, `prepare_inputs.py` (where present), `evaluate_decision.py` and `evaluate_laya_suite.py`: earlier single-question experiment utilities. They are not the default joint recipe and their budgets and metrics must not be mixed with it.

See [THIRD_PARTY.md](THIRD_PARTY.md) for schema attribution. Dataset access and licenses remain the caller's responsibility.
