# GKD

GKD（Generalized Knowledge Distillation，广义知识蒸馏）训练算法由论文 [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/pdf/2306.13649) 提出。该算法通过结合离线（off-policy）和在线（on-policy）学习策略，将教师模型的知识迁移到学生模型中。

## 损失函数

当给定输入序列 $x$ 与输出序列 $y$，GKD 的损失函数可以写为：

$$
\mathcal{L}_{\text{GKD}}(x, y) = \sum_{t=1}^{|y|} D(P_{\text{teacher}}(\cdot | x, y_{<t}), P_{\text{student}}(\cdot | x, y_{<t}))
$$

其中：
- $y_{<t} = (y_1, y_2, \ldots, y_{t-1})$：前 $t-1$ 个 token 的序列
- $P_{\text{teacher}}(\cdot | x, y_{<t})$：教师模型在给定上下文 $x, y_{<t}$ 时的输出概率分布
- $P_{\text{student}}(\cdot | x, y_{<t})$：学生模型在给定上下文 $x, y_{<t}$ 时的输出概率分布
- $D(\cdot, \cdot)$：散度函数，用于度量两个概率分布之间的差异性

## 散度度量函数

### KL 散度（Kullback-Leibler Divergence）

KL 散度是衡量两个概率分布 $P$ 和 $Q$ 之间差异的非对称度量：

$$
\text{KL}(P \| Q) = \sum_v P(v) \log \frac{P(v)}{Q(v)} = \mathbb{E}_{v \sim P}\left[\log \frac{P(v)}{Q(v)}\right]
$$

### Forward KL 与 Reverse KL

在知识蒸馏中，根据 KL 散度中两个分布的顺序不同，有两种选择：

#### Forward KL（前向 KL）

$$
\text{KL}(P_{\text{student}} \| P_{\text{teacher}}) = \sum_v P_{\text{student}}(v) \log \frac{P_{\text{student}}(v)}{P_{\text{teacher}}(v)}
$$

**特性**：Mode-seeking（寻模）
- 期望在学生分布下计算
- 学生模型倾向于集中在教师模型的峰值区域（高概率区域）

#### Reverse KL（反向 KL）

$$
\text{KL}(P_{\text{teacher}} \| P_{\text{student}}) = \sum_v P_{\text{teacher}}(v) \log \frac{P_{\text{teacher}}(v)}{P_{\text{student}}(v)}
$$

**特性**：Mode-covering（覆模）
- 期望在教师分布下计算
- 学生模型倾向于覆盖教师的整个分布（包括低概率区域）

### 广义 Jensen-Shannon 散度（Generalized JSD）

GKD 使用广义 JSD 作为核心度量，通过参数 $\beta \in [0, 1]$ 在 Forward KL 和 Reverse KL 之间进行**平滑插值**。

对于两个概率分布 $P$ 和 $Q$，广义 JSD 定义为：

$$
D_{\text{JSD}(\beta)}(P, Q) = \beta \cdot \text{KL}(P \| M) + (1-\beta) \cdot \text{KL}(Q \| M)
$$

其中混合分布 $M$ 定义为：

$$
M = \beta \cdot P + (1-\beta) \cdot Q
$$

- 当 $\beta = 0.5$ 时，退化为标准的对称 JSD
- 通过调节 $\beta$，可以在 Mode-seeking 和 Mode-covering 之间权衡

在 GKD 中，我们令 $P = P_{\text{teacher}}$，$Q = P_{\text{student}}$，因此：

$$
D_{\text{JSD}(\beta)}(P_{\text{teacher}}, P_{\text{student}}) = \beta \cdot \text{KL}(P_{\text{teacher}} \| M) + (1-\beta) \cdot \text{KL}(P_{\text{student}} \| M)
$$

其中 $M = \beta \cdot P_{\text{teacher}} + (1-\beta) \cdot P_{\text{student}}$

> 对极端情况（$\beta = 0$ 或 $\beta = 1$），直接计算单个 KL 散度：
> - 当 $\beta = 0$ 时：直接定义 $D = \text{KL}(P_{\text{teacher}} \| P_{\text{student}})$（Reverse KL，Mode-covering）
> - 当 $\beta = 1$ 时：直接定义 $D = \text{KL}(P_{\text{student}} \| P_{\text{teacher}})$（Forward KL，Mode-seeking）
> - 当 $0 < \beta < 1$ 时：使用上述混合分布公式进行插值

通过调节 $\beta$ 参数，可以在不同的散度度量之间进行插值，当 $\beta = 0.5$ 时，散度为标准的对称 JSD。

## 三种训练模式

GKD训练具有三种训练模式，区别在于输出序列 $y$ 的来源。

### 模式选择逻辑

训练时，每个样本按照以下优先级选择模式：

```python
# 伪代码：模式选择逻辑
if random() < lmbda:
    # Mode 1: On-Policy 学习，由学生模型采样输出序列
    y = student.generate(x)
    source = "student"
elif seq_kd:
    # Mode 2: Sequential KD，由教师模型采样输出序列
    y = teacher.generate(x)
    source = "teacher"
else:
    # Mode 3: Off-Policy 学习，使用数据集中的输出序列
    y = y_ground_truth
    source = "dataset"

# 相同的损失函数
loss = D_JSD(P_teacher(·|x,y), P_student(·|x,y))
```

### Mode 1: On-Policy 学习
设置参数`lambda`, 以概率 $\lambda$ 触发，使用学生模型采样 $y \sim P_{\text{student}}(\cdot | x)$

- 学生模型从**自己生成的序列**中学习
- 暴露在自己可能犯的错误中，学会**自我纠正和错误恢复**
- 对齐训练分布与推理分布
- 提升模型的鲁棒性和实际应用表现

**适用场景**：
- 学生模型已有一定生成能力
- 希望提升模型在真实推理场景下的表现

### Mode 2: Sequential KD（`seq_kd=True` 且未触发 on-policy）
设置参数 `seq_kd=True`, 当未触发 on-policy 时，使用教师模型采样

**数据来源**：$y \sim P_{\text{teacher}}(\cdot | x)$

### Mode 3: Off-Policy 学习（其他情况）

**数据来源**：$y = y^* \sim \text{Dataset}$

- 学生模型从**数据集的标注序列**中学习


## 参数设置

我们可以通过设置以下参数进行 GKD 训练：

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|--------|---------|------|
| `--teacher_model` | str | 必需 | - | 教师模型路径或模型 ID |
| `--beta` | float | 0.5 | [0.0, 1.0] | 散度插值系数<br>• 0.0: Reverse KL (覆模，更多样)<br>• 0.5: JSD (平衡，**推荐**)<br>• 1.0: Forward KL (寻模，更专注) |
| `--lmbda` | float | 0.5 | [0.0, 1.0] | On-Policy 学习触发概率<br>• 0.0: 纯 Off-Policy<br>• 0.5: 混合策略 (**推荐**)<br>• 1.0: 纯 On-Policy |
| `--seq_kd` | bool | False | True/False | 是否使用教师生成序列<br>• False: 非 on-policy 时使用数据集<br>• True: 非 on-policy 时使用教师生成 |
| `--temperature` | float | 0.9 | > 0 | 生成采样温度，控制随机性 |
| `--max_completion_length` | int | 512 | > 0 | 生成时的最大 token 数 |
