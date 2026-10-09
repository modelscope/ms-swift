## 数据-树状层级

+ level1 count=2
	+ `<APPEAL> / <QUESTION>`
+ level2 count=4
	+ `<APPEAL> [Scene] 主站 <APPEAL> / <QUESTION> [Scene] 主站 <QUESTION> / <QUESTION> [Scene] 其他主站 <QUESTION> / <QUESTION> [Scene] 非主站 <QUESTION>`
+ level3 count=200+
	+ appeal_example: `<APPEAL> [Scene] 主站 | [Cat_lv1] 主站-活动 | [Cat_lv2] 暑期活动 <APPEAL>`
	+ faq_example: `<QUESTION> [Scene] 主站 | [Cat_lv1] 主站-直播支付 | [Cat_lv2] 其他诉求 <QUESTION>`
+ level4 count=5k+
    + appeal_example: `"<APPEAL> [Scene] 主站 | [Cat_lv1] 活动 | [Cat_lv2] 暑期活动 | [appeal_description] 用户参与暑期活动时遇到的各类问题，包括设备绑定限制、奖励发放规则、商品配送及库存问题等，需要客服提供准确的活动规则解释和解决方案。 | [symptom] 用户在参与暑期活动过程中遇到设备绑定限制、励未按时发放、商品配送异常等问题 | [user_goal] 获取活动规则的准确解释，解决参与活动时的技术障碍，确保活动奖励顺利领取 | [system_action] 根据用户具体问题，按照活动规则提供标准化解答，包括设备绑定说明、奖励发放流程、商品配送解决方案等 | [FAQ] ['百天心愿之旅-为什么助力提示“您已经助力过了”', '百天心愿之旅- 如何添加小组件', '百天心愿之旅-如何参与活动'] <APPEAL>",`
level4层级数据，包含树状分类信息 + 实际用于召回的文档
## 目标
构建综合损失函数，infonce用于主要用于embedding空间学习，分类用于主要用于分类能力学习

## 1) 训推一致性
别做“前缀匹配”，要做“真正的分类头”
A. “前缀/字符串匹配去算 loss”（不推荐）
- 做法：把 `"<QUESTION> [Scene] 主站 | ..."` 当作文本前缀/目标串，让模型“生成/匹配”它来算 LM loss。
- 问题：
    - **这训练的是生成式对齐，不是分类能力**；你线上如果是用 embedding 检索，等于训推目标错位。
    - label 很长、噪声大（字段多、描述长），LM loss 会被“复述字段”主导，而不是学“路由边界”。
B. “接 softmax 分类头”（推荐）
- 做法：在 query embedding 上接一个或多个线性层（level1/2/3），用 **Cross-Entropy**（可加 label smoothing）训练。
- 好处：
    - **训推一致**：线上要用路由就直接用 logits/prob；线上不用路由也能当正则项。
    - 分类目标可控（类别数、mask、层级一致性约束）。
> 你这类“树状标签 + 检索”任务，分类头是更干净的 inductive bias。

---

## 2) 性能：
不需要“分开推理两次”，一次 forward 够

如果你用的是“共享 encoder + 多头输出”，线上流程可以是：

1. **一次 forward** 得到：
    - query embedding (e_q)
    - (p(\text{lv2}\mid q))、(p(\text{lv3}\mid q))（可选）
        
2. 向量检索时用 **metadata filter / 分片索引**：
    - 过滤到预测子树（或 top-k 子树）再 ANN search
    - embedding 不用重算
        
所以**模型推理只跑一次**。额外开销主要是
- 多几个线性层（基本可忽略）
- 可能多做 1~2 次 ANN search（比如 top-2 lv2 各搜一次），这比再跑一遍 encoder 便宜得多。
    

---

## 3) conditional softmax / path-KL 是什么

### Conditional softmax（层级条件 softmax / top-down 概率分解）

它的核心是：**每个父节点对自己的孩子做一个 softmax**，叶子类概率是一路条件概率的乘积：

[  
q_y(\theta)=\prod_{u\in A(y)\setminus{r}} r_u(\theta),\quad  
r_{C(y)}(\theta)=\mathrm{softmax}_{C(y)}(\theta_{C(y)})  
]

这在 NeurIPS 2022 的层级分类论文里有明确表述（引用了 YOLO-9000 的 conditional softmax）。

> 工程意义：**天然保证层级一致性**（预测到某叶子，路径上的父类也隐含成立），也便于你做“先 lv2 再 lv3”的路由。

⚠️ 但也要注意：这篇 NeurIPS 论文的实验里，conditional softmax 在不少 setting 下不如“平铺 softmax”强（尤其 seen 类），所以它更像是“结构约束/一致性优先”的选择。

---

### Path-KL（Tree-path KL divergence / 路径一致性 KL）

这类 loss 的思路是：把 **各层级的标签 one-hot 拼接成一个大分布**，把各层 head 的输出也拼接后做 log-softmax，然后用 KL 去对齐，从而**惩罚层级不一致**。

你提到的 “path-KL” 在一个分层分类工作里被明确写成：拼接各层 labels 与 outputs 后用 KL 对齐，并把它作为额外项加入总 loss。([arXiv](https://arxiv.org/html/2406.11608v2 "Visually Consistent Hierarchical Image Classification"))

> 工程意义：比“只加多层 CE”更强调“路径整体正确”，对你这种树状标签（2/4/200+）更贴近最终目标（别把 lv3 预测到别的 lv2 子树里）。

---

## 4) 给你两套“默认配置”（按线上是否用路由）

你的场景里，最关键的分歧就是：

- **(a) 线上用分类做路由（先路由到子树再检索）**
    
- **(b) 分类只做训练正则，线上不用**
    

我直接给两套可落地的默认配置，你选一套跑 ablation 即可。

---

### 配置 A：线上用路由（推荐优先尝试）

**线上：** top-k 路由 → 子树检索 → 合并候选 → rerank（可选）

**训练 loss：**

[  
\mathcal{L}  
=\mathcal{L}_{\text{InfoNCE}}  
+\lambda\sum_{l=1}^{3}\alpha_l ,\mathrm{CE}_l  
+\beta,\mathcal{L}_{\text{PathKL}}  
]

**建议权重（默认起跑点）：**

- (\lambda=0.15)
    
- ((\alpha_1,\alpha_2,\alpha_3)=(0.2,1.0,0.5)) （优先把 lv2 路由训稳）
    
- (\beta=0.05)
    
- CE 用 label smoothing：0.05
    

**路由策略（默认）：**

- lv2：top-2（或 top-1 但加 fallback）
    
- 置信度门槛：若 (p(\text{lv2 top1})<0.45) ⇒ **全库检索兜底**
    
- 检索：每个命中的 lv2 子索引取 top-200，合并后再 rerank / 再按向量相似度取 top-N
    

**负例层级（减少 false negative 的默认规则）：**

- 70% negatives：来自 **不同 lv2**
    
- 25% negatives：同 lv2 不同 lv3（hard，但**下采样**）
    
- 5% negatives：同 lv3（尽量别放，极易把“同义不同写法”打成假负例）
    

并且打开 fake negative mask（下面 Swift 变量里有）。

---

### 配置 B：线上不用路由（只做正则）

**训练：** 仍然多头分类，但权重更轻，避免损伤 embedding 空间。

- (\lambda=0.05)
    
- ((\alpha_1,\alpha_2,\alpha_3)=(0.2,0.7,0.3))
    
- (\beta=0.02)
    

线上直接全库 ANN + rerank；分类 logits 只用于监控/分析 badcase。

---

**改造思路（最小侵入）：**

1. 用 Swift 的 embedding pipeline 把 batch 里的 query/pos/neg 都 encode 出 embedding（或复用它的 encode 逻辑）
2. 在 query embedding 上加 3 个线性头：lv1/2/3 logits
3. `compute_loss`：
    
    - 先算 (\mathcal{L}_{\text{InfoNCE}})（你现在已有）
        
    - 再算多层 CE（必要时做 conditional softmax / mask）
        
    - 再算 path-KL（可选）
        
    - 按上面配置加权求和
        

**分类 loss 的构建规则（你直接照这个实现就行）：**

- lv1：2 类（APPEAL/QUESTION）→ CE
    
- lv2：4 类 → CE（这是路由主力）
    
- lv3：200+ 类 → CE（建议 label smoothing；必要时 class weight）
    
- conditional softmax：lv3 的 softmax **只在真实 lv2 的子集内**（训练时）；推理时在预测 lv2 的子集内（或 top-k lv2 各算一次）


下面这个“**InfoNCE（检索几何） + 分类（树结构监督）**”的混合训练，在你的**树状层级标签**（type/scene/cat）场景里，**经常能带来更稳的层级一致性与更好的粗粒度路由**；但要避免把 embedding 空间“压成类别原型”，导致**同类内部细粒度区分变差**。

---

## 1) 对树状层级数据：InfoNCE + 分类混合，通常能带来什么

### 更可能变好（你的场景很匹配）

1. **层级一致性更强**
   分类 loss 会明确告诉模型：`<APPEAL>/<QUESTION>`、`Scene`、`Cat_lv1/lv2` 是“应当可分”的结构信号；对比学习只靠“相似/不相似”对，容易在**同父兄弟**、**父子节点**附近摇摆。

2. **减少 false negative 的副作用**（尤其你混了多层级负例时）
   树里语义相近节点被当负例会互相拉扯；分类 head 会提供“全局方向”，让表示空间更不容易被这些近邻负例搞乱。

3. **可直接服务线上：先分类做路由/候选剪枝，再做向量检索**
   即使分类不完美，也可以用 top-k 类别做“软路由”，再在子树内做 embedding 检索。

> 这种“CE + 对比”联合目标在多篇工作里被用来提升表示或分类稳定性（对比项提供几何约束，CE 提供判别监督）。([arXiv][1])

### 可能变差的点（需要规避）

* **同类内部区分能力下降**：CE 会倾向把表示聚成类中心（甚至出现“类顶点”式结构），如果 CE 权重太大，会牺牲检索中同类 FAQ/子类的排序质量。([arXiv][2])

**经验结论**：对“以检索为主”的任务，建议让 **InfoNCE 做主任务**，分类做 **辅助（小权重 / 只训部分层级 / 分离投影头）**。

---

## 2) 分类损失怎么构建：最常用的 3 套规则（适配你的 level1/2/3）

你现在有：

* level1: 2 类（`<APPEAL>/<QUESTION>`)
* level2: 4 类（scene 细分）
* level3: 200+（leaf 类别，如 Cat_lv2）

### 方案 A（最推荐）：**多头 level-wise CE（共享 encoder）**

同一个 query embedding (h) 上接 3 个 softmax 头：

* (p^{(1)}=\mathrm{softmax}(W_1 h)) 预测 level1
* (p^{(2)}=\mathrm{softmax}(W_2 h)) 预测 level2
* (p^{(3)}=\mathrm{softmax}(W_3 h)) 预测 level3（leaf）

总 loss：
$$
\mathcal{L}
===========

\mathcal{L}*{\text{InfoNCE}}
+
\lambda \sum*{l=1}^{3} \alpha_l , \mathcal{L}^{(l)}_{\text{CE}}
$$

其中
$$
\mathcal{L}^{(l)}*{\text{CE}}
= -\sum*{k=1}^{K_l} y^{(l)}_k \log p^{(l)}_k
$$

**实践建议**：

* (\alpha_l) 让粗粒度更大、细粒度更小（例如 (\alpha_1=1,\alpha_2=0.7,\alpha_3=0.3)），避免 leaf CE 把空间压扁。
* (\lambda) 从 0.05～0.3 起步调（检索优先）。
* 类别长尾明显时：对 CE 用 **class weight / focal loss / logit adjustment**（至少做 class weight）。

### 方案 B：**Top-down（条件分类 / masked softmax）**

先预测父节点，再在其子集中预测子节点（训练时可用 ground-truth 父节点做 mask）：

* level3 的 softmax 只在“正确 level2 的子类集合”上归一化

优点：天然保证层级一致；缺点：实现复杂、推理链路更长。层级分类文献里这是常见思路之一。([arXiv][3])

### 方案 C：**路径多标签（BCE）**

把路径上所有节点都当作 label（每个节点一个二分类），用 BCE：

* 易保证“祖先必须为真”
* 但 label 数会变大（尤其节点多时），工程/采样更重

---

## 3) 在 ms-swift 里怎么做：两条落地路径（从省事到彻底）

> 先说约束：Swift 的 embedding 训练格式目前 **每条样本只能 1 个 positive**（与你前面讨论一致）。([swift.readthedocs.io][4])
> 另外：把额外字段（比如 `label_lv1/lv2/lv3`）传到 loss，需要 `remove_unused_columns=False`，这样它们会进入 `compute_loss`。([swift.readthedocs.io][5])

### 路径 1（最实用、改动最小）：**自定义 Trainer.compute_loss（embedding task 内加 CE）**

思路：

1. 仍然用 `--task_type embedding --loss_type infonce` 走 Swift 的 embedding 数据管道
2. 数据里新增 `label_lv1,label_lv2,label_lv3`
3. 设置 `--remove_unused_columns false`
4. 继承 Swift/Transformers 的 Trainer，override `compute_loss`：

   * 调模型得到 query/pos/neg embedding → 计算 InfoNCE
   * 用 query embedding (h) 走 classification heads → 计算 CE
   * 按权重相加返回

Swift 文档明确支持把“额外列”透传到 `compute_loss` 以便自定义 loss。([swift.readthedocs.io][5])
Swift 也给出了 CLI 训练入口（`sft_main/TrainArguments`），你可以写一个自定义训练脚本复用其参数体系。([swift.readthedocs.io][6])

**关键点**：

* **分类 head 建议接在 encoder 输出 (h)**（而不是对比投影后的 (z)），并与对比 head 分离，减小互相伤害。
* InfoNCE 常见实现会构造 batch 内相似度矩阵，显存随 batch 近似 (O(B^2)) 增长（你调 batch 时要注意）。([GitHub][7])

### 路径 2（工程更稳）：**两阶段/交替训练（不用改 Swift 内部）**

* Stage A：只训 embedding（InfoNCE）
* Stage B：冻结/半冻结 encoder，小步训分类 head（或联合微调但小 lr）
* 可选：再回 Stage A 做少量对比微调

如果你主要想要“分类用于路由”，这种最稳，不容易把 embedding 空间训坏。

---






## 外加分类任务的核心思路：给“树的边界/路由”一个**显式监督**，让 embedding 的几何别被近邻负例拉扯坏

你现在用 InfoNCE 做检索，本质是在学一个**相似度空间**：让 (q) 靠近对应 (d^+)，远离 (d^-)。  
但你又有一个很强的结构先验：**文档在树上（level1/2/3）**，很多“近邻”其实是**同父兄弟 / 父子**，语义很近、容易被当 hard negative。此时加一个分类目标相当于告诉模型：

- “这些边界（type/scene/cat）是需要可分的”
    
- “同一子树内部允许更近，不要因为采样到相似负例就把空间推得乱七八糟”
    

这就是“InfoNCE（局部相对关系） + 分类（全局判别边界）”的互补。

---

## 你说的实现方式对：用 embedding 向量接 softmax 头做 CE（但注意用哪个向量）

### Qwen3-Embedding 的向量通常就是 EOS hidden state

Qwen3 Embedding 技术报告明确写了：在输入末尾加 `[EOS]`，**最终 embedding 取最后一层对应 `[EOS]` 的 hidden state**。([arXiv](https://arxiv.org/pdf/2506.05176 "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models"))  
官方博客也同样描述为取最终 `[EOS]` token 的 hidden state。([Qwen](https://qwenlm.github.io/blog/qwen3-embedding/ "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | Qwen"))

所以你说的“取 eos hiddenstate 做向量，然后做 softmax 分类头”在工程上是成立的。

### 但更推荐：分类头接在 **pre-projection / pre-normalize** 的表示上

常见做法是把 encoder 输出记为 (h)（未归一化的表征），检索用的向量是 (z=\mathrm{norm}(P(h)))。  
分类 logits 用 (h) 而不是 (z)：

- 检索希望“角度/距离结构”稳定（常配合 L2-normalize）
    
- 分类希望“线性可分”即可，不一定要受 normalize 约束
    

这样可以降低“分类把 embedding 压成类中心”对检索排序的副作用。

---

## “独热分类是不是意味着语义无用？”——不是这样理解

你说的“独热”只是在说 **监督信号的形式**（label 是离散 id），**并不意味着输入语义没用**。原因：

1. **分类头学习的是“用语义特征把类别分开”**  
    标准分类器形式是：encoder (\phi(x)) 输出表示 (h)，再接线性层 (W) 得到 logits，然后做 softmax + cross-entropy。Graf 等人总结的就是这种经典结构 (f=\arg\max \circ W \circ \phi)。([arXiv](https://arxiv.org/pdf/2102.08817 "Dissecting Supervised Contrastive Learning"))  
    标签是 one-hot，但要把不同类分开，模型只能依赖输入文本的语义线索（词、短语、句法、领域表达）来形成可分的表示。
    
2. **“标签语义”确实没有被直接利用，但“文本语义”在驱动学习**  
    CE 没有显式告诉模型“类A和类B相近”，它只知道“这条样本属于类A”。  
    如果你希望“类之间也有语义关系”（树结构就是关系），可以通过：
    

- 分层/条件 softmax（只在父类子集内归一化）
    
- path 一致性正则（如把路径概率与真值路径对齐的 KL 类 loss）
    

来把“树的语义结构”编码进监督里（你之前问的 path-KL 就是这类思想）。([Emergent Mind](https://www.emergentmind.com/topics/tree-path-kl-divergence-tp-kl?utm_source=chatgpt.com "Tree-Path KL Divergence (TP-KL)"))（概念综述）

---

## 你这个树状检索任务里，分类任务最正确的用法（训推一致）

### 关键：分类一定要对 **query 表示**做（别对 doc 文本“读标签”做）

你的 doc 文本里已经写了 `[Scene] 主站 | [Cat_lv1] ...`，如果你对 doc 表示做分类，模型可能“读到标签就完事”，监督价值很低。

**推荐：**

- 输入：用户 query（不带这些标签字段）
    
- label：这条 query 对应的正例 doc 的 (lv1, lv2, lv3) id
    
- 分类头：3 个 head（2 类 / 4 类 / 200+ 类）
    

### loss 形式（默认）

设 query 表示为 (h_q)，检索向量为 (z_q)，doc 向量为 (z_d)：

$$[  
\mathcal{L}_{\text{InfoNCE}}(z_q, z_d)  
+  
\lambda\sum_{l\in{1,2,3}}\alpha_l ,\mathrm{CE}(W_l h_q, y_l)  
]$$

**建议起跑权重（检索优先）：**

- (\lambda = 0.1)
    
- ((\alpha_1,\alpha_2,\alpha_3) = (0.2, 1.0, 0.3))  
    （lv2 “路由层”更重要；lv3 类多且长尾，别压太重）
    

**长尾处理（lv3 200+）：**

- 优先用 class weight 或者 label smoothing（0.05）
    
- 只有当头部类碾压严重再考虑 focal
    

---

## 一句话总结

- 是的：实现上就是“取 EOS hidden（更推荐用 pre-projection (h)）→ 多个 softmax head → CE”。([arXiv](https://arxiv.org/pdf/2506.05176 "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models"))
    
- 不是：one-hot label 不代表语义无用；语义体现在 encoder 如何把文本映射到可分的表示空间，分类头只是提供边界监督。([arXiv](https://arxiv.org/pdf/2102.08817 "Dissecting Supervised Contrastive Learning"))
    
- 对你：分类应当监督 **query** 的层级标签（路由边界），别让模型从 doc 文本里“读标签”作弊。
    

如果你把你现在训练时的 `query 文本格式`（是否带 instruction）和 `doc 文本是否固定含 Scene/Cat` 贴一下，我可以把上面这套变成一个更具体的“输入模板 + 哪些字段要 mask/保留”的配置。

---

## 最终落地版本（2026-03-02）

### 已实现能力

- 新增混合损失：`route_hybrid_infonce`（主任务 InfoNCE + 辅助分类）
- 支持 3 种模式：
  - `ce3`: 3 层 CE（lv1/lv2/lv3）
  - `ce3_conditional`: lv3 使用基于 lv2 子树的 conditional softmax CE
  - `ce3_conditional_pathkl`: 在上一模式基础上增加 Path-KL
- 分类特征优先读取 `pre_norm_last_hidden_state`，不存在则自动回退 `last_hidden_state`
- embedding 输出新增：
  - `pre_norm_last_hidden_state`
  - `pre_projection_last_hidden_state`（与 pre-norm 同值别名）
- `Seq2SeqTrainer.compute_loss` 已透传：
  - `label_lv1`
  - `label_lv2`
  - `label_lv3`

### 默认超参

- `ce3`: `lambda=0.10`, `alpha=(0.2,1.0,0.3)`, `beta=0`
- `ce3_conditional`: `lambda=0.15`, `alpha=(0.2,1.0,0.5)`, `beta=0`
- `ce3_conditional_pathkl`: `lambda=0.15`, `alpha=(0.2,1.0,0.5)`, `beta=0.05`

### 关键环境变量

- `ROUTE_CLS_MODE`: `ce3|ce3_conditional|ce3_conditional_pathkl`
- `ROUTE_CLS_LAMBDA`
- `ROUTE_CLS_ALPHA`（例如 `0.2,1.0,0.5`）
- `ROUTE_CLS_BETA_PATHKL`
- `ROUTE_CLS_LABEL_SMOOTHING`
- `ROUTE_CLS_FEATURE_KEY`（默认 `pre_norm_last_hidden_state`）
- `ROUTE_HIERARCHY_JSON`（conditional/path-kl 模式必需）
- 可选：
  - `ROUTE_NUM_LV1`
  - `ROUTE_NUM_LV2`
  - `ROUTE_NUM_LV3`
  - `ROUTE_CLS_MISSING_LABEL_STRATEGY`（`error|skip`）

### 层级映射 JSON schema（建议）

```json
{
  "num_lv1": 2,
  "num_lv2": 4,
  "num_lv3": 200,
  "lv2_to_lv1": [0, 0, 1, 1],
  "lv3_to_lv2": [0, 0, 1, 1, 2, 3]
}
```

### 训练启动示例

```bash
export ROUTE_CLS_MODE=ce3_conditional_pathkl
export ROUTE_HIERARCHY_JSON=/abs/path/route_hierarchy.json
export ROUTE_CLS_FEATURE_KEY=pre_norm_last_hidden_state

swift sft \
  --external_plugins /Users/liujunchen/WorkSpace/Frames/ms-swift/workSpace/plugin/route_cls_loss_plugin.py \
  --loss_type route_hybrid_infonce \
  --remove_unused_columns false \
  ...
```
