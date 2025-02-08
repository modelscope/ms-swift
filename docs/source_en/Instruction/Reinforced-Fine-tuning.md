# Reinforced Fine-Tuning

Reinforced fine-tuning is one of the most important functionalities in current model training, with various implementations. SWIFT has already supported the atomic capabilities required for reinforced fine-tuning, such as sampling, reinforcement learning, and fine-tuning. Currently, we provide a specific example of rejection sampling fine-tuning, which can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/rft/rft.py).

## Concept of Reinforced Fine-Tuning

The concept of reinforced fine-tuning has been proposed since 2022 (or even earlier). Its general workflow typically includes the following steps:

1. Generate data using a specific model or augment the original dataset.
2. Train the target model using the generated data.
3. Repeat the above process if necessary.

**Step 1:**

- If the data-generating model is a larger model, such as GPT, Qwen-Max, DeepSeek-V3/R1, etc., this process can be understood as distillation.
- If the data-generating model is the same model being trained, this can be considered self-improvement fine-tuning.
- If the sampling process involves sampling a batch, fitting the data with KL divergence and rewards, and iterating continuously, it can be classified as on-policy algorithms like PPO or GRPO.
- Sampling algorithms include Monte Carlo sampling, do_sample, group beam search, DVTS, etc.
- The sampling process can incorporate ORM (Outcome Reward Model), PRM (Process Reward Model), diversity filtering, language filtering, etc.

**Step 2:**

- If SFT (Supervised Fine-Tuning) is used, it is referred to as rejection sampling fine-tuning.
- If reinforcement learning is used, it is called reinforcement learning fine-tuning.

**Step 3:**

- If distillation is performed using a larger model (e.g., Monte Carlo sampling distillation with a larger model), the process usually does not involve iterations.
- If the same model is used for sampling or algorithms like PPO are applied, iterations are typically included.

In general, the common approaches to reinforced fine-tuning include:

1. **Distillation**: Sampling high-quality data in bulk from a larger model using methods like Monte Carlo or do_sample, and training a smaller model on this data.
2. **Self-improvement**: Sampling a portion of high-quality data from the same model, filtering it, and training the model iteratively.
3. **On-policy RL**: Using methods like PPO or GRPO for iterative training.

The sampling process is usually much more time-consuming than the training process. If data is distilled using GPT or other large models, token costs must be considered. Thus, reinforced fine-tuning is generally a supplementary mechanism for fine-tuning, except for special cases like DeepSeek-R1.

DeepSeek-R1 uses the GRPO algorithm to enable the emergence of CoT (Chain-of-Thought) capabilities from scratch in a base model. This method requires large-scale cluster support and sufficiently large models for capability emergence. This is not discussed in detail here, but more information can be found in the [paper analysis](https://zhuanlan.zhihu.com/p/19714987272).

Some related papers on reinforced fine-tuning:

- Rejection Sampling Fine-Tuning: https://arxiv.org/pdf/2308.01825
- ReST: https://arxiv.org/pdf/2308.08998
- B-STAR: https://arxiv.org/pdf/2412.17256
- DeepSeekMath: https://arxiv.org/pdf/2402.03300
- Qwen-Math-PRM: https://arxiv.org/pdf/2501.07301
- DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1/tree/main

## When to Use Reinforced Fine-Tuning

Since LLaMA3, we have observed a very noticeable yet rarely mentioned phenomenon: when training an Instruct model using a CoT-enabled training dataset and evaluating it on the corresponding test set, the test set performance tends to degrade. For example, training `llama3.1-8b-instruct` on the GSM8K training set and evaluating the generated checkpoint on the test set reveals performance degradation.

This phenomenon mainly arises from the issue of knowledge forgetting disaster in models. During fine-tuning by model manufacturers, a significant amount of CoT data is often included. When solving mathematical tasks, the model's capability often originates not from the math dataset itself but potentially from datasets like ARC. This inference is supported by [some works](https://zhuanlan.zhihu.com/p/19269451950). Continued training on general tasks disrupts the model's existing capabilities, leading to performance degradation.

However, it is always correct to prioritize fine-tuning. Fine-tuning allows the model to quickly adapt to the dataset distribution at a low cost. Reinforced fine-tuning should be used under the following conditions:

1. The model has already been fine-tuned but does not meet the requirements.
2. Stronger CoT capabilities are needed.
3. Base model training for general capabilities is necessary, and the original dataset no longer improves performance.
4. The output results for corresponding queries can be relatively accurately evaluated, such as tasks with clear results (math, code) or clear processes (translation, style fitting).

Reinforced fine-tuning heavily depends on the accuracy of reward evaluations. If the evaluations are inaccurate, the training may oscillate without progress or even degrade the model performance.

## SWIFT Implementation

SWIFT supports the `sample` command, which is used for model sampling. Currently supported sampling methods include:

- **do_sample**: A sampling method for open-source models; future updates will include support for model distillation.
  - URL sampling will also be supported in the future for large-model distillation.

- **mcts**: Monte Carlo sampling, currently under review, with future support planned.
- **dvts**: Currently under investigation.

We have provided a general [RFT script](https://github.com/modelscope/ms-swift/tree/main/examples/train/rft/rft.py). This script supports self-improvement training and allows dynamic adjustments of sampling temperature, PRM thresholds, and other hyperparameters. The training method is flexible (e.g., fine-tuning, DPO) and supports iterative retraining of the original model or continued training from the previous iteration, even loading all training states from the previous iteration. Developers can incorporate additional data filtering (e.g., ensuring rows with the same ID come from the same query), including diversity checks, language filtering, etc.

## Experimental Results

We used the RFT script to train and evaluate the `competition_math` dataset in the math domain. The results are as follows:

| Model                      | MATH Score | Training Method | Iterations | Post-Training MATH Score  |
|----------------------------|------------|-----------------|------------|---------------------------|
| LLaMA3.1_8b               | 12.0       | SFT             | 3          | 25.2 (LLaMA3.1_8b_sft)   |
| LLaMA3.1_8b_sft           | 25.2       | RFT             | 2          | 32.4                     |
| LLaMA3.1_8b_instruct      | 52.2       | SFT             | 2          | 39.0                     |
| LLaMA3.1_8b_instruct      | 52.2       | RFT             | 3          | 58                       |
| Qwen2.5_math_7b_instruct  | 79.6       | RFT             | 2          | 83.2                     |

As shown, applying SFT to the `competition_math` dataset resulted in significant performance degradation for the instruct model. However, RFT improved the model's capabilities, even for the state-of-the-art `Qwen2.5_math_7b_instruct` math model.

Specifically, we tested the GSM8K metric for `Qwen2.5_math_7b_instruct`:

| Model                      | GSM8K Score | Post-RFT GSM8K Score |
|----------------------------|-------------|-----------------------|
| Qwen2.5_math_7b_instruct  | 92.8        | 91.6                 |

As shown, RFT training did not significantly change the GSM8K score, avoiding the previously mentioned performance degradation phenomenon.

## Future Roadmap

1. More sampling methods，MCTS for example
2. Distill from super huge model
3. On policy RFT like PPO
