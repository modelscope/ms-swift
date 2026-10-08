# Intern-Decision-4B 联合字段训练复现

## 章节一 模型简介

从 Qwen3.5-4B 指令基座训练语言主干全参数，冻结视觉和 projector。沿用 Intern-Decision 的联合 JSON 骨架与 <decision> 标记，答案只进入 labels，标记前一个位置预测该字段答案；使用全词表交叉熵。本阶段是文字训练，未验收图像样本和跨样本 packing。

## 章节二 性能和精度数据

数值、权重和原始预测只本地保存。训练数据为公开 Typed Decisions 的固定切分：960训练案例、120验证、120校准，官方400测试案例保持独立。每案例5个字段，训练4800个监督目标。原始官方训练集未公开，因此这是方法与框架迁移复现，不保证重训达到官方七项精度。

Historical validation on the pinned reproduction revisions completed: four-NPU smoke and full-state resume, 120 optimizer steps, final model export/reload, joint-field evaluation and all seven accuracy suites. Aggregate accuracy remained below the official target. This PR does not claim reproduction of the unpublished training corpus or official model quality. Final metrics and raw predictions remain local. The rebased PR revision requires its own device validation; historical evidence is not a new-head training run.

## 章节三 复现分支和指导

本框架个人分支为 `intern_decision_4b_npu_pr` (PR); `intern_decision_4b_repro` (validated historical source)，继承此前 `intern_decision_4b_npu` 的已验证 NPU 接入，新增联合字段数据检查与训练入口。SWIFT 基线为 e815cb65efcb27c00c3cc5524603d6916ab5a112，Twinkle 基线为513041163495cdfef71f8b4f90edeee9e1694201；两个框架分别训练，不加载另一框架训练后的权重。

基础镜像：`quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1`，不导出镜像。独立容器、四张分配的910B卡，模型只读挂载到 `/models/Qwen3.5-4B`，源码挂载到 `/workspace/framework`，数据到 `/data/joint`。Python3.12、Torch2.10.0+cpu、torch_npu2.10.0.post2、Transformers5.15.1；Twinkle独立虚拟环境固定peft0.19.0。纯Python修改，无新增C++编译。

## 章节四 启动和配置参数

先使用 `prepare_joint.py --prepared PATH --output NEW_PATH --official-schema OFFICIAL_SCHEMA` 编译固定案例，核对官方schema、目标不泄漏及切分互斥。官方schema引用Intern-Decision源码，SHA256为 d2fce8ffb8af19dca08a04eb01f4daae70ed43311d73c052ddcab797346d79bd。输出目录必须不存在。

新建容器命令（`RUNTIME` 先创建为空目录；两个框架使用不同容器和运行目录，同一组卡串行执行）：

```bash
mkdir -p "$RUNTIME"
bash create_container.sh swift "$CONTAINER" "$RUNTIME" "$CHECKOUT" "$MODEL" "$DATA"
# Twinkle 时将 swift 改为 twinkle；DATA 下包含 joint 和 prepared。
```

在容器内的示例目录执行：

```bash
bash run_joint.sh smoke
bash run_joint.sh resume
bash run_joint.sh train

ASCEND_RT_VISIBLE_DEVICES=0 /workspace/.venv/bin/python evaluate_joint.py \
  --checkpoint /workspace/outputs/train/checkpoint-120 \
  --data /data/prepared/test.jsonl --output /workspace/results/joint-test
```

正式预算为2个epoch、全局batch16案例，即120个optimizer更新；学习率2e-6、cosine最低2e-7、warmup0.03、AdamW weight_decay0.01、seed42。禁止截断、关闭跨样本packing。正式运行固定最终权重，不用test挑选checkpoint。smoke/resume保存模型、优化器、调度器及随机状态；正式阶段只保存最终模型，不能用其宣称恢复全部训练状态。

最终权重使用当前分支的 `evaluate_joint.py`（两框架采用相同的 MS-SWIFT HF NPU BF16 推理后端）按联合字段在固定validation/test上评测；官方七项使用独立推理评测环境及固定官方输入，不把训练loss当精度。训练性能统计有效更新耗时、峰值显存和监督token，不使用推理压测冒充训练性能。

## 章节五 问题列表

- 旧单字段数据编译会丢失联合上下文，本分支保留完整案例并验证官方输入一致。
- Twinkle外部CE不做HF式shift，labels必须提前移动一次；SWIFT保留未shift标签，禁止双移位。
- 联合字段的验证CE分母必须是有效监督token数，不能继续按案例数除。
- 多卡checkpoint使用既有CPU同步及优化器分片保存逻辑，需实际恢复检查。
- 原始训练数据和完整生产配方缺失；当前公开业务集不能覆盖全部官方七项任务。

## 章节六 总结

本分支复现联合决策训练语义与双框架执行流程；正式训练和精度状态以实际运行记录为准。推理 eager/graph 分支与本训练分支分离。
