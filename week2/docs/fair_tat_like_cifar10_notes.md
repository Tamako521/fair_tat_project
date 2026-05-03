# FAIR-TAT-like CIFAR-10 复现实验说明

## 本阶段目标

本阶段目标是先跑通一个接近 FAIR-TAT 思想的 CIFAR-10 轻量复现入口，为后续完整复现做准备。当前版本不宣称完全等同于 FAIR-TAT 原文，而是实现其中最关键的一条主线：根据类别混淆情况选择 targeted PGD 的目标类别，并用这些对抗样本进行训练。

参考资料：

- arXiv:2410.23142，FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training
- ar5iv HTML 版本：https://ar5iv.labs.arxiv.org/html/2410.23142

## 方法理解

FAIR-TAT 关注 adversarial training 中的类别级不公平问题。普通 untargeted adversarial training 往往提升整体鲁棒性，但不同类别的 clean accuracy 和 robust accuracy 差距可能变大，弱类别会更吃亏。

FAIR-TAT 的核心变化是使用 targeted adversarial training。目标类别不是固定选择，也不是完全随机选择，而是根据 class-wise false positive score 构造采样分布。某个类别如果经常被模型错误预测为目标类别，说明该类别对应的决策区域更容易吸引其他类别样本，因此训练时更频繁地生成朝该类别移动的 targeted adversarial examples。

## 当前实现

本项目当前实现了 FAIR-TAT-like 版本：

1. 第 1 个 epoch 使用 clean warmup，让模型先有基本分类能力。
2. 每轮评测 clean test set，得到 confusion matrix。
3. 将 confusion matrix 的对角线置零，按列统计 false positive count。
4. 加 smoothing 后归一化，得到 target prior。
5. 训练 targeted PGD adversarial examples 时，对每个样本从 target prior 中采样目标类别，并排除真实标签。
6. 训练损失使用 clean loss 与 adversarial loss 的加权和：

```text
loss = (1 - adv_weight) * CE(model(x), y)
     + adv_weight * CE(model(x_adv_targeted), y)
```

其中 `x_adv_targeted` 是朝采样目标类别生成的 targeted PGD 样本，但训练目标仍然是原始真实标签 `y`。

## 当前输出指标

新入口会输出：

- clean accuracy
- clean worst-class accuracy
- FGSM robust accuracy
- FGSM worst-class robust accuracy
- targeted PGD original accuracy
- targeted PGD worst-class robust accuracy
- targeted PGD target success rate
- class-wise robust accuracy
- target prior 历史

## 与原文仍有差距的部分

当前版本先实现 target class prior 和 targeted PGD 训练主线，暂未实现：

- 原文中更完整的训练区间动态更新策略。
- 基于 robust accuracy 的 per-class perturbation margin 自适应。
- PRN-18、XCiT-S12 等更强 backbone。
- CIFAR-100、STL-10、ISIC 2019 等数据集扩展。
- AutoAttack、Square Attack、common corruptions 等更强评测。

## 后续计划

下一步建议在当前入口上扩大训练规模，例如 `train-size=5000` 或 `10000`，并与第一周 clean baseline、FGSM AT、固定目标 targeted PGD AT 做对比，重点观察 worst-class robust accuracy 是否改善。
