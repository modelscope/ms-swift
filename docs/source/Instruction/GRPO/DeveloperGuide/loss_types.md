# Loss Types

GRPO训练支持多种不同的loss类型，主要区别在于归一化的维度和梯度处理方式上有所不同。

## 损失函数

token 级别上，GRPO 训练使用以下损失函数

$$\mathcal{L}_{i,t} = -\min\left(\rho_{i,t} A_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_{i,t}\right)$$

当设置`loss_type cispo`时，使用 cispo 损失

$$\mathcal{L}_{i,t}^{\text{CISPO}} = -\text{detach}\left(\min(\rho_{i,t}, \epsilon_{\text{high}})\right) \cdot A_{i,t} \cdot \log \pi_\theta(y_{i,t}|y_{i,<t})$$

当设置`loss_type sapo`时，使用软门控替代硬裁剪，详见 [SAPO](../AdvancedResearch/SAPO.md)

$$\mathcal{L}_{i,t}^{\text{SAPO}} = -g_{i,t} \cdot A_{i,t}$$

其中 $g_{i,t} = \sigma(\tau \cdot (\rho_{i,t} - 1))$ 是温度控制的软门控函数。

其中：
- $\rho_{i,t} = \frac{\pi_\theta(y_{i,t}|y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|y_{i,<t})}$ 是重要性采样权重
- $A_{i,t}$ 是优势函数
- $\epsilon$ 和 $\epsilon_{\text{high}}$ 是clipping参数
- $\text{detach}(\cdot)$ 表示该项不参与梯度计算
- $\sigma(\cdot)$ 是 sigmoid 函数，$\tau$ 是温度参数

## GRPO

`--loss_type grpo`

GRPO是标准的损失函数实现，对每个样本的token-level损失取平均，然后对所有样本取平均。

**公式：**

$$\mathcal{L}_{\text{GRPO}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}$$

其中：
- $N$ 是批次中的样本数量
- $T_i$ 是第$i$个样本的completion token数量

**归一化维度：** 样本维度（先对每个样本的所有token取平均，再对所有样本取平均）

## BNPO (Batch Normalized Policy Optimization)

`--loss_type bnpo`

BNPO将所有样本的所有token的损失直接求和，然后除以所有completion token的总数量。

**公式：**

$$\mathcal{L}_{\text{BNPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{\sum_{i=1}^{N} T_i}$$

其中：
- $N$ 是批次中的样本数量
- $T_i$ 是第$i$个样本的completion token数量

**归一化维度：** Token维度（对所有completion token取平均）

## DR-GRPO

`--loss_type dr_grpo`

DR-GRPO将所有样本的所有token的损失求和，然后除以批次大小乘以最大completion长度。

**公式：**

$$\mathcal{L}_{\text{DR-GRPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{N \times L_{\text{max}}}$$

其中：
- $N$ 是批次中的样本数量
- $T_i$ 是第$i$个样本的completion token数量
- $L_{\text{max}}$ 是最大completion长度

**归一化维度：** 固定维度（批次大小 × 最大completion长度）

## CISPO

`--loss_type cispo`

CISPO损失按所有进程的completion token总数进行归一化。

**公式：**

$$\mathcal{L}_{\text{CISPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}^{\text{CISPO}}}{\sum_{\text{all processes}} \sum_{i=1}^{N_p} T_{p,i}}$$

其中：
- $N$ 是当前进程批次中的样本数量
- $T_i$ 是第$i$个样本的completion token数量
- $N_p$ 是第$p$个进程的样本数量

**归一化维度：** 全局token维度（跨所有进程的completion token总数）

## DAPO

`--loss_type dapo`

DAPO与BNPO类似，使用token-level归一化，但基于全局数据（多进程）进行归一化。

**公式：**

$$\mathcal{L}_{\text{DAPO}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{T_i} \mathcal{L}_{i,t}}{\sum_{\text{all processes}} \sum_{i=1}^{N_p} T_{p,i}}$$

其中：
- $N$ 是当前进程批次中的样本数量
- $T_i$ 是第$i$个样本的completion token数量
- $N_p$ 是第$p$个进程的样本数量

**归一化维度：** 全局token维度（跨所有进程的completion token总数）

## SAPO

`--loss_type sapo`

SAPO使用温度控制的软门控替代硬裁剪，实现平滑的梯度衰减。归一化方式与GRPO相同。

详细说明请参考 [SAPO](../AdvancedResearch/SAPO.md)
