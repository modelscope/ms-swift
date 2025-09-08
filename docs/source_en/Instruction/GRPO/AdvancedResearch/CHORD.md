# On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD)

**Version Requirement**: ms-swift>=3.9

This document describes the CHORD algorithm proposed in the paper [On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting](https://arxiv.org/abs/2508.11408). The core idea of CHORD is to dynamically integrate expert data (SFT) into reinforcement learning by a dual control mechanism: a global weight μ plus a token-level weight φ, thereby balancing imitation and exploration.

## Algorithm Overview
CHORD mixes training by introducing the SFT loss into the GRPO loss. The overall objective is:

$$
    \mathcal{L}_{\text{CHORD}} = (1 - \mu) \cdot \mathcal{L}_{\text{GRPO}} + \mu \cdot \mathcal{L}_{\text{SFT}}
$$

where:
- $\mathcal{L}_{\text{GRPO}}$: on-policy RL loss based on on-policy samples (similar to PPO).
- $\mathcal{L}_{\text{SFT}}$: supervised fine-tuning (SFT) loss.
- $\mu \in [0, 1]$: global balancing coefficient that controls the contribution of the SFT signal to the overall gradient.

### Configuration (data and batch sizes)
We can implement CHORD training based on GRPO training.

CHORD requires specifying an additional SFT dataset and batch size at training time:
- `chord_sft_dataset`: the SFT dataset that provides expert data.
- `chord_sft_per_device_train_batch_size`: SFT mini-batch size per device.

---

## Two CHORD Variants

The paper proposes two variants: CHORD-μ and CHORD-φ.

### CHORD-μ
CHORD-μ gradually decays μ during training to transition from imitating experts toward autonomous exploration.

Parameters:
- `chord_mu_peak`: the peak value of μ.
- `chord_mu_valley`: the final decayed value of μ.
- `chord_mu_warmup_steps`: number of training steps to ramp μ up to the peak.
- `chord_mu_decay_steps`: number of training steps to decay μ from peak to valley.

### CHORD-φ (Token-level weighting)
CHORD-φ uses a token-wise weighting function φ to dynamically control each expert token's gradient contribution.

Definition of φ:
$$
    \phi(y_t^\star, \pi_\theta) = p_t \cdot (1 - p_t)
$$

where:
- $p_t = \pi_\theta(y_t^\star \mid x, y_{<t}^\star)$ is the model's current predicted probability of the expert token.
- When $p_t \approx 0.5$ (model uncertainty), φ is maximal → emphasize tokens the model is uncertain about.
- When $p_t \approx 0$ or $p_t \approx 1$, φ → 0 → avoid overemphasizing tokens that are already certain or impossible.

Parameter to enable φ weighting:
- `chord_enable_phi_function: bool = False`
  - Set to `True` to enable token-wise weight φ.

Note: If using a constant μ, set `chord_mu_peak` and `chord_mu_valley` to the same value.

<details>
<summary>Code implementation of μ scheduling and loss computation</summary>
See the `GRPOTrainer` method `_compute_chord_loss`.
</details>

Training reference script: https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/chord.sh
