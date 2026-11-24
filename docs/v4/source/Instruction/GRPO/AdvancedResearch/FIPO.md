# FIPO: Future-KL Influenced Policy Optimization

作者： [li2zhi](https://github.com/li2zhi)

[FIPO](https://arxiv.org/abs/2603.19835) 是一种面向长链推理的 value-free RL 方法。它保留 GRPO/DAPO 的整体训练框架，但改变 token 级策略更新的加权方式：不再让一个序列级 advantage 均匀作用到所有 token，而是用折扣累积的 Future-KL 信号判断“从当前 token 开始的后续轨迹”整体是在被增强还是被削弱。

## 核心思想

GRPO/DAPO 中，每个 response 的 token 通常共享同一个序列级 advantage：

$$
\hat{A}_{i,t} = \hat{A}_{i}
$$

这种做法稳定且简单，但 credit assignment 粒度较粗。FIPO 引入当前策略与旧策略在每个 token 上的 log-prob shift：

$$
\Delta \log p_t = \log \pi_\theta(y_t \mid x, y_{<t}) -
\log \pi_{\mathrm{old}}(y_t \mid x, y_{<t})
$$

如果 $\Delta \log p_t > 0$，说明当前训练正在提高该 token 的概率；如果小于 0，则说明该 token 正在被压低。FIPO进一步从当前位置向后折扣累积该信号：

$$
\mathrm{FutureKL}_t =
\sum_{k=t}^{T}\gamma^{k-t} M_k \Delta \log p_k
$$

其中 $M_k$ 是 completion mask，$\gamma = 2^{-1 / \text{decay\_rate}}$。`decay_rate` 越大，越远的 future token 对当前位置的影响越强；`decay_rate` 越小，Future-KL 越偏局部。然后将 Future-KL 映射为 influence weight：

$$
f_t = \mathrm{clip}(\exp(\mathrm{FutureKL}_t), 1-\epsilon_f, 1+\epsilon_f)
$$

最终把原本的 advantage 改成 future-aware advantage：

$$
\tilde{A}_{i,t} = \hat{A}_{i} \cdot f_{i,t}
$$

## 参数

| 参数                        | 类型      | 默认值    | 说明                                                                                                             |
|---------------------------|---------|--------|----------------------------------------------------------------------------------------------------------------|
| `--loss_type`             | `str`   | `grpo` | 设置为`fipo` 启用 FIPO loss                                                                                         |
| `--delta`                 | `float` | `None` | 启用后会同时用于 Future-KL 高 IS ratio token 过滤和主 loss 的 dual-clip 上限，应大于 `1 + epsilon_high`，对齐FIPO 32B训练脚本建议设置为 `10.0` |
| `--fipo_decay_rate`       | `float` | `32.0` | Future-KL 折扣半衰参数，实际折扣为`2 ** (-1 / fipo_decay_rate)`                                                            |
| `--fipo_clip_range`       | `float` | `0.2`  | influence weight 裁剪范围；`0.2` 表示默认裁剪到 `[0.8, 1.2]`                                                               |
| `--fipo_clip_high_only`   | `bool`  | `true` | 若为`true`，权重只裁剪到 `[1.0, 1.0 + fipo_clip_range]`，更偏向放大正 Future-KL                                                |
| `--fipo_safety_threshold` | `float` | `4.0`  | 负 advantage 且 IS ratio 超过该阈值时，将 FIPO 权重限制到 `[0.8, 1.0]` 以避免过度惩罚                                                |

## 训练示例

[swift](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/fipo.sh)
