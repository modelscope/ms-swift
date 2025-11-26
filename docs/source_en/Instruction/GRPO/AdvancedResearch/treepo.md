# TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling

**Version Requirement**: ms-swift>=3.11

## Principle Introduction
[TreePO paper](https://arxiv.org/abs/2508.17445) proposes a tree-structured modeling method. This method organizes sequence generation into a segmented tree structure search. Through dynamic branching, backtracking, and early termination mechanisms, it significantly improves the reuse rate of the key-value cache, thereby reducing computational overhead, while maintaining or even enhancing the diversity of exploration.

![TreePO Overview](../../../../resources/treepo.png)

## Implementation Details
[TreePO implementation example](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/treepo/tree_rollout_plugin.py), which references the [official implementation](https://github.com/multimodal-art-projection/TreePO/blob/main/recipe/treepo/vllm_rollout_tree.py) provides sample code for a TreePO training plugin，covering logic related to multi-round interactions, termination judgment, and branch rollback.

**Note:** In actual use, you need to rewrite the logic of methods such as step and check_finished according to your own scenario requirements to ensure that they can execute as expected in the custom scenario.

The complete training script can be found at [script](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/treepo/tree_rollout.sh).

## Test Data
| /                       | batch size | num generation | max_tree_deep | total_infer | saving ratio |
| ----------------------- | ---------- | -------------- | ------------- | ----------- | ------------ |
| original implementation | 8          | 8              | 4             | 5965        | 0.00%        |
| tree(max_divergence=3)  | 8          | 8              | 4             | 3678        | 38.34%       |
|                         |            |                |               |             |              |
| original implementation | 8          | 8              | 5             | 4312        | 0.00%        |
| tree(max_divergence=2)  | 8          | 8              | 5             | 2513        | 52.69%       |
| tree(max_divergence=3)  | 8          | 8              | 5             | 2990        | 30.66%       |
|                         |            |                |               |             |              |
| original implementation | 8          | 8              | 6             | 5202        | 0.00%        |
| tree(max_divergence=2)  | 8          | 8              | 6             | 3348        | 35.64%       |
| tree(max_divergence=3)  | 8          | 8              | 6             | 3888        | 25.26%       |
