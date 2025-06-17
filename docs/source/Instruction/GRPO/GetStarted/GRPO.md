# GRPO

[GRPO(Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) 算法利用组内相对优势计算来替代 PPO 算法中独立的价值模型，并直接在损失函数中加入 KL 散度惩罚来提高训练稳定性。

<img src="../../../../resources/PPOverseGRPO.png" alt="PPO/GRPO算法比较" width="600" />

