# FAIR-TAT 方法对齐笔记

## 论文信息

- 论文：FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training
- 作者：Tejaswini Medi, Steffen Jung, Margret Keuper
- arXiv：2410.23142
- 当前 arXiv 页面显示该工作最早提交于 2024-10-30，后续版本继续更新。
- 参考来源：https://arxiv.org/abs/2410.23142

## 核心问题

传统 adversarial training 通常优化平均鲁棒性，但会造成类别之间鲁棒性差异扩大。FAIR-TAT 关注 class-wise robustness 和 worst-class robustness，目标是让困难类别在 clean 和 adversarial 条件下都不要被明显牺牲。

## Target Class 选择策略

FAIR-TAT 的关键不是随机选择 targeted attack 的目标类别，而是使用 class-wise false positive score 构造 target prior：

1. 统计模型把其他类别误判为某一类别的次数。
2. 某个类别被误判进入的频率越高，说明它在当前模型边界中具有更强吸引性或混淆性。
3. 训练时更频繁地把 targeted adversarial examples 指向这些高 false positive 类别。
4. 采样 target label 时排除真实标签，避免 target 与 ground truth 相同。

当前项目中的实现对应：

- `src.target_selection.false_positive_prior`
- `src.target_selection.sample_target_labels`
- `src.training.train_one_epoch_fair_tat_like`

## 当前实现与原文差距

当前实现已经完成轻量 FAIR-TAT-like 路线：

- 使用 clean confusion matrix 估计 false positive prior。
- 每轮训练后更新 target prior。
- 使用 targeted PGD 生成对抗样本。
- 使用 clean loss 与 adversarial loss 加权训练。

仍需继续补齐：

- 更细粒度的 interval-level target prior 更新，而不是只在 epoch 级更新。
- 引入 per-class robust accuracy，进一步调节 target prior 或 class loss weight。
- 与原文中 CFA / TRADES / FAT 组合的方式继续对齐。
- 用更强模型和更强攻击评测确认现象是否稳定。

## 第三周落地动作

本周先完成工程侧扩展：

1. 加入 ResNet18 模型入口。
2. 加入 untargeted PGD 和 multi-step FGSM 评测。
3. 使用最佳参数进行多 seed 复现实验。
4. 记录 class-wise robust accuracy 与 worst-class robust accuracy。

per-class 调节策略暂时作为下一步增强，不直接声称已经完整复现原文。
