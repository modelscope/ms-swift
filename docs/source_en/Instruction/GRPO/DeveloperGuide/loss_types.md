# Loss Types

GRPO training supports multiple loss types, with the main differences being the normalization dimension and gradient handling.

## Loss Function

At the token level, GRPO training uses the following loss function:

$$\mathcal{L}_{i,t} = -\min\left(\rho_{i,t} A_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_{i,t}\right)$$

When setting `loss_type cispo`, the CISPO loss is used:

$$\mathcal{L}_{i,t}^{\text{CISPO}} = -\text{detach}\left(\min(\rho_{i,t}, \epsilon_{\text{high}})\right) \cdot A_{i,t} \cdot \log \pi_\theta(y_{i,t}|y_{i,<t})$$

When setting `loss_type sapo`, soft gating replaces hard clipping, see [SAPO](../AdvancedResearch/SAPO.md)

$$\mathcal{L}_{i,t}^{\text{SAPO}} = -g_{i,t} \cdot A_{i,t}$$

where $g_{i,t} = \sigma(\tau \cdot (\rho_{i,t} - 1))$ is the temperature-controlled soft gate function.

where:
- $\rho_{i,t} = \frac{\pi_\theta(y_{i,t}|y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|y_{i,<t})}$ is the importance sampling weight
- $A_{i,t}$ is the advantage function
- $\epsilon$ and $\epsilon_{\text{high}}$ are the clipping parameters
- $\text{detach}(\cdot)$ indicates that this term does not participate in gradient computation
- $\sigma(\cdot)$ is the sigmoid function, $\tau$ is the temperature parameter

## GRPO

`--loss_type grpo`

GRPO is the standard loss function implementation that averages the token-level losses for each sample, then averages across all samples.

**Formula:**

$$\mathcal{L}_{\text{GRPO}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}$$

where:
- $N$ is the number of samples in the batch
- $T_i$ is the number of completion tokens for the $i$-th sample

**Normalization Dimension:** Sample dimension (first average over tokens for each sample, then average over all samples)

## BNPO (Batch Normalized Policy Optimization)

`--loss_type bnpo`

BNPO sums all token losses from all samples and then divides by the total number of completion tokens.

**Formula:**

$$\mathcal{L}_{\text{BNPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{\sum_{i=1}^{N} T_i}$$

where:
- $N$ is the number of samples in the batch
- $T_i$ is the number of completion tokens for the $i$-th sample

**Normalization Dimension:** Token dimension (average over all completion tokens)

## DR-GRPO

`--loss_type dr_grpo`

DR-GRPO sums all token losses from all samples and then divides by the batch size multiplied by the maximum completion length.

**Formula:**

$$\mathcal{L}_{\text{DR-GRPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{N \times L_{\text{max}}}$$

where:
- $N$ is the number of samples in the batch
- $T_i$ is the number of completion tokens for the $i$-th sample
- $L_{\text{max}}$ is the maximum completion length

**Normalization Dimension:** Fixed dimension (batch size × maximum completion length)

## CISPO

`--loss_type cispo`

CISPO loss is normalized by the total number of completion tokens across all processes.

**Formula:**

$$\mathcal{L}_{\text{CISPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}^{\text{CISPO}}}{\sum_{\text{all processes}} \sum_{i=1}^{N_p} T_{p,i}}$$

where:
- $N$ is the number of samples in the current process batch
- $T_i$ is the number of completion tokens for the $i$-th sample
- $N_p$ is the number of samples for the $p$-th process

**Normalization Dimension:** Global token dimension (total completion tokens across all processes)

## DAPO

`--loss_type dapo`

DAPO is similar to BNPO, using token-level normalization, but based on global data (multi-process) normalization.

**Formula:**

$$\mathcal{L}_{\text{DAPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{\sum_{\text{all processes}} \sum_{i=1}^{N_p} T_{p,i}}$$

where:
- $N$ is the number of samples in the current process batch
- $T_i$ is the number of completion tokens for the $i$-th sample
- $N_p$ is the number of samples for the $p$-th process

**Normalization Dimension:** Global token dimension (total completion tokens across all processes)

## SAPO

`--loss_type sapo`

SAPO uses temperature-controlled soft gating instead of hard clipping to achieve smooth gradient attenuation. The normalization method is the same as GRPO.

For details, please refer to [SAPO](../AdvancedResearch/SAPO.md)
