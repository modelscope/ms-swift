# DAPO


[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)在GRPO的基础上设置了几种trick，分别是
- [Clip Higher](#clip-higher)
- [Dynamic Sampling](#dynamic-sampling)
- [Token level Loss]()
- [Overlong Filtering]()
- [Soft Overlong Punishment](#soft-overlong-punishment)

## Clip Higher
PPO和GRPO使用对称裁剪范围(如±0.2)限制策略更新幅度，虽然保证了稳定性，但也制约了模型的探索能力。特别是当某些token在旧策略中概率极低时，即使当前梯度显示其应被强化(A>0)，最大增幅也被严格限制。

DAPO使用非对称裁剪范围, 提高上裁剪范围来鼓励模型进行探索：

- 上界(鼓励侧)放宽至0.28
- 下界(抑制侧)保持0.2不变

GRPO中，默认使用`epsilon`设置用对称裁剪范围

使用参数

- `epsilon_high` 设置上裁剪范围，此时参数`epsilon` 为下裁剪范围

## Dynamic Sampling
GRPO对每个问题采样多个回答计算组间优势，

$$
\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$
而当生成的所有输出{oi}获得相同奖励时，组间优势等于0，会出现梯度消失导致训练效率下降

DAPO引入动态采样策略解决这一问题：

- 采样阶段跳过组间奖励标准差为0的数据
- 持续生成样本直到填满批次


使用参数

- `dynamic_sample true` 来开启动态采样
- `max_resample_times` 设置最多重采样次数

## Token level Loss
GRPO 在归一化损失时采用句子级归一化，这会导致损失计算具有长度偏差。

DAPO 使用token级归一化，避免了回答长度在损失计算上的偏差。

使用参数

- loss_type bnpo 来使用token级归一化


## Overlong Filtering
DAPO 认为被强制截断的回复的奖励噪声较大，可能会导致模型难以区分质量问题和长度问题。为此，DAPO 筛除了训练中被截断的数据，使其不参与损失计算。

使用参数

- overlong_filter 开启对超长样本的过滤


## Soft Overlong Punishment
语言模型常面临生成长度控制难题：

- 过长输出可能被截断，导致正确内容被误判

- 无约束生成长度影响实用性和计算效率

DAPO 设计了三段式长度惩罚函数：

$$
R_{\text{length}}(L) =
\begin{cases}
0, & \text{if } L \leq L_{\text{cache}} \\[10pt]
-\dfrac{L - L_{\text{cache}}}{L_{\text{max}} - L_{\text{cache}}}, & \text{if } L_{\text{cache}} < L < L_{\text{max}} \\[10pt]
-1, & \text{if } L \geq L_{\text{max}}
\end{cases}
$$

在长度位于(L_cache < L < L_max)区间时设置线性递增惩罚，在(L ≥ L_max)时设置最大惩罚(-1)


使用参数

- `reward_funcs soft_overlong` 启用该奖励函数
- `soft_max_length` 设置L_max，默认等于为模型的最长输出长度(max_completion_length)。
- `soft_cache_length`: 设置L_cache

## 参数设置
综上所述，我们可以基于GRPOTrainer，设置以下参数实现 DAPO 训练。

| 参数                 | 类型      | 值      |
|----------------------|-----------|-------------|
| `--loss_type`        | `str`     | `bnpo`     |
| `--epsilon_high`     | `float`   | `0.28`      |
| `--dynamic_sample`   | `bool`    | `true`      |
| `--max_resample_times` | `int`   | `3`        |
| `--overlong_filter`  | `bool`    | `true`      |
| `--reward_funcs`     | `str`     | `soft_overlong`|
| `--soft_cache_length`     | `int`     | `4096`|
