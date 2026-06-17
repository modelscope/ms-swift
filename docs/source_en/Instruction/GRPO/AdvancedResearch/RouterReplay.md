# Router Replay (R2/R3)

**TL;DR**: In RL training of MoE models, routing inconsistency between the training engine and the inference engine can significantly amplify training-inference mismatch, and even cause training collapse. Router Replay eliminates this inconsistency by replaying fixed routing masks during the training forward pass. Depending on the replay source, there are two strategies: R2 (Vanilla Routing Replay) and R3 (Rollout Routing Replay).

## Background

### Three Policies in MoE RL

In GRPO training of MoE models, there are three distinct policy stages that share the same model weights but may differ in routing behavior:

| Policy | Notation | Routing Result | Description |
|--------|----------|---------------|-------------|
| **Training Policy** | $\pi_\theta$ | $e^{\pi}_t$ | The model during gradient updates |
| **Old Policy** | $\pi_{\theta_{\text{old}}}$ | $e^{\pi}_{\text{old},t}$ | The model state before batch updates |
| **Rollout Policy** | $\mu_{\theta_{\text{old}}}$ | $e^{\mu}_{\text{old},t}$ | The sampling policy in the inference engine (e.g., vLLM), with the same weights as old policy, but different routing due to kernel implementation differences, precision, etc. |

Here, $\pi_{\theta_{\text{old}}}$ and $\mu_{\theta_{\text{old}}}$ have identical weights at sampling time, but due to implementation differences between the inference and training engines (e.g., operator implementations), routing results may differ even for the same input.

### Decomposition of Training-Inference Mismatch

According to the [paper](https://arxiv.org/abs/2507.18071), the token-level importance sampling ratio can be decomposed into two factors:

$$
\frac{\pi_\theta(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}{\mu_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{training-inference discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x, y_{<t})}}_{\text{policy staleness}}
$$

For MoE models, expert routing is deeply coupled with both factors:

- **Training-inference discrepancy**: Routing inconsistency between the training and inference engines ($e^{\pi}_{\text{old},t} \neq e^{\mu}_{\text{old},t}$) amplifies output divergence
- **Policy staleness**: As mini-batch updates proceed, routing also drifts ($e^{\pi}_t \neq e^{\pi}_{\text{old},t}$), further deviating from the sampling policy

## R2: Vanilla Routing Replay

The core idea of R2 is to **replay the routing determined by the old policy in the training engine** ($e^{\pi}_{\text{old},t}$) during gradient updates.

### Principle

During the training forward pass, first run a forward pass with the old policy weights to record the expert indices selected by each MoE Router layer, then force the training model $\pi_\theta$ to use these indices in its forward pass:

$$
g_{\text{replay},i} = \frac{I^{\pi}_{\text{old},i} \cdot \exp(s_{\text{train},i})}{\sum_j I^{\pi}_{\text{old},j} \cdot \exp(s_{\text{train},j})}
$$

where $I^{\pi}_{\text{old}}$ is the old policy's routing mask and $s_{\text{train}}$ is the router logits computed by the training model. The softmax still operates on the training logits, so gradients can flow back to the router weights normally.

### Properties

| Scenario | Behavior |
|----------|----------|
| **First mini-batch** (on-policy) | $\theta = \theta_{\text{old}}$, so $e^{\pi}_t = e^{\pi}_{\text{old},t}$, **target policy unchanged** (no bias) |
| **Subsequent mini-batches** (off-policy) | $\theta \neq \theta_{\text{old}}$, so $e^{\pi}_t \neq e^{\pi}_{\text{old},t}$, **target policy changed** (biased), but policy staleness is controlled |

## R3: Rollout Routing Replay

The core idea of R3 is to **replay the routing determined by the rollout policy in the inference engine** ($e^{\mu}_{\text{old},t}$) during the training forward pass.

### Principle

During sampling in the inference engine (e.g., vLLM), additionally record the expert indices (routing mask) for each token at every MoE Router layer, then pass these masks to the training engine and force $\pi_\theta$ to use them in its forward pass:

$$
g_{\text{replay},i} = \frac{I^{\mu}_{\text{old},i} \cdot \exp(s_{\text{train},i})}{\sum_j I^{\mu}_{\text{old},j} \cdot \exp(s_{\text{train},j})}
$$

### Properties

| Scenario | Behavior |
|----------|----------|
| **All mini-batches** | $e^{\mu}_{\text{old},t} \neq e^{\pi}_t$, **target policy always changed** (always biased) |

R3 **simultaneously addresses** two problems:
- **Training-inference discrepancy**: The routing used on the training side = the routing from the inference side
- **Policy staleness**: Routing is fixed and does not drift with $\theta$

R3 reduces the MoE training-inference KL divergence to **near Dense model levels**. In mathematical reasoning RL tasks, training without R3 collapses after 60-120 steps, while R3 training remains stable throughout.

## R2 vs R3: Recommendations

The GSPO paper provides the following conclusions from large-scale controlled experiments (Qwen3-30B-A3B, hundreds of thousands of GPU hours):

| Training Scenario | Recommendation | Reason |
|-------------------|---------------|--------|
| **On-policy** (gbs = mbs) | Neither needed | $\theta = \theta_{\text{old}}$, policy staleness ≈ 0, IS correction suffices |
| **Mildly off-policy** (gbs = 2×mbs) | **R2** | No bias in the first mini-batch, sufficient to stabilize training |
| **Heavily off-policy** (gbs ≥ 4×mbs) | **R3** | Staleness too large, must simultaneously correct training-inference mismatch |

> **Core trade-off**: R2 is more conservative (only corrects intra-engine inconsistency, smaller bias); R3 is more aggressive (eliminates all routing inconsistency at the source, but always changes the target policy, introducing bias). When off-policiness is small, R3's bias cost outweighs its benefit; when off-policiness is large, the cost of not correcting training-inference mismatch far exceeds the bias.

### Compatibility with Other Methods

- R3 is **orthogonal** to **GSPO** and can be combined for further improvement
- R3 combined with **TIS** may not provide additional gains (R3 already eliminates inconsistency at the source; TIS's additional correction may be redundant)
- Router Replay and **Clipping** are both essential in off-policy training

## Router Mask Caching

The R3 paper also proposes that routing masks can be cached alongside the KV Cache: for the same prefix tokens, the MoE Router output is deterministic, so routing masks can be stored and reused together with the prefix KVCache. This is particularly important in multi-turn Agent scenarios (tool calling), avoiding the need to re-prefill the prefix to obtain routing masks, with an overall rollout latency overhead of less than 3%.

## Swift Implementation

### Parameters

Select the routing replay strategy via the `--router_replay_mode` parameter:

| Value | Description |
|-------|-------------|
| `disabled` (default) | No routing replay |
| `R2` | Vanilla Routing Replay: record old policy routing in the training engine and replay |
| `R3` | Rollout Routing Replay: export routing masks from the inference engine and replay in training |

Environment requirements:

- R3 requires vLLM ≥ 0.14.0 to support returning `routed_experts` information.
- Router Replay is currently only available with the Megatron backend, requiring megatron-core ≥ 0.16.0.

### Relationship with Training-Inference Correction

Router Replay and the importance sampling (IS) correction described in [Training-Inference Mismatch](training_inference_mismatch.md) are complementary:

- **IS correction**: Corrects probability divergence at the loss level via weighting
- **Router Replay**: Eliminates the source of divergence at the model architecture level by fixing routing

## References

1. [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)
2. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
3. [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
4. [Megatron Core Router Replay Design Document](https://docs.nvidia.com/megatron-core/developer-guide/nightly/api-guide/router_replay.html)
