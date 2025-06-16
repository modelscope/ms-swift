# DAPO


[Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476)在GRPO的基础上设置了几种trick，分别是
- Clip Higher
- Dynamic Sampling
- Overlong Filtering
- Token level Loss
- Soft Overlong Punishment

以上trick，我们可以基于GRPOTrainer，设置以下参数实现。

其中Token level Loss是通过使用参数 loss type `bnpo` 实现

| 参数                 | 类型      | 值      |
|----------------------|-----------|-------------|
｜`--loss_type`        | `str`      | `bnpo`     |
| `--epsilon_high`     | `float`   | `0.28`      |
| `--dynamic_sample`   | `bool`    | `true`      |
| `--overlong_filter`  | `bool`    | `true`      |
| `--reward_funcs`     | `str`     | `soft_overlong`|
| `--max_resample_times` | `int`    | `3`        |
