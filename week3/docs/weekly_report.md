# 周报：FAIR-TAT-like 稳定性复现与强评测扩展

姓名：郭家棋

## 一、本周完成工作

本周主要在第二周 FAIR-TAT-like CIFAR-10 复现基础上继续扩展，重点是确认最佳参数稳定性、加入更强评测攻击，并开始尝试更强模型结构。

1. 继续对齐 FAIR-TAT 原文，整理了 target class 选择策略和 class-wise fairness 目标。
2. 在当前最佳参数 `adv_weight=0.7, prior_smoothing=10.0, PGD-5` 上完成两个 full CIFAR-10 随机种子复现实验。
3. 新增 multi-step FGSM 和 untargeted PGD 评测，补充比 FGSM / targeted PGD 更严格的鲁棒性观察。
4. 新增 ResNet18 for CIFAR-10 模型入口，并完成小规模与 5k probe 实验。
5. 训练脚本增加每轮 checkpoint 保存，避免断电或中断导致阶段结果完全丢失。
6. 初步调研 ISIC 2019 数据集，整理了疾病标签、年龄、性别、解剖部位等公平性属性。

## 二、实验顺序

1. `stage01_resnet18_smoke`：验证 ResNet18、新攻击评测和日志保存是否跑通。
2. `stage02_best_seed7_full_strong_eval`：使用最佳参数在 full CIFAR-10 上做 seed 7 复现实验。
3. `stage03_best_seed123_full_strong_eval`：使用最佳参数在 full CIFAR-10 上做 seed 123 复现实验。
4. `stage04_resnet18_5k_probe`：使用 ResNet18 在 5k CIFAR-10 子集上做初步训练验证。

## 三、主要实验结果

主要设置：

- 数据集：CIFAR-10
- 训练攻击：targeted PGD
- 新增评测攻击：multi-step FGSM、untargeted PGD
- `epsilon=8/255`
- `alpha=2/255`
- 设备：CPU

| 实验 | 模型 | 训练规模 | clean acc | clean worst | FGSM robust | IFGSM robust | untargeted PGD robust | targeted PGD original | targeted PGD worst |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage02 seed7 | SmallCNN | full | 0.6332 | 0.3280 | 0.2271 | 0.1816 | 0.1694 | 0.4717 | 0.2533 |
| stage03 seed123 | SmallCNN | full | 0.6480 | 0.3464 | 0.2461 | 0.1997 | 0.1826 | 0.4956 | 0.2374 |
| stage04 ResNet18 probe | ResNet18 | 5k | 0.4470 | 0.0899 | 0.1582 | 0.1895 | 0.1641 | 0.3828 | 0.0784 |

## 四、结果分析

1. 当前最佳参数在两个 full CIFAR-10 随机种子上都能稳定跑通，clean accuracy 约为 `0.63-0.65`。
2. targeted PGD original accuracy 在两个 seed 上分别为 `0.4717` 和 `0.4956`，说明第二周筛出的参数具有一定稳定性。
3. untargeted PGD-10 明显比 targeted PGD 评测更严格，robust accuracy 下降到约 `0.17-0.18`，worst-class 指标仍偏低。
4. ResNet18 入口已经跑通，但 5k 数据和 2 epoch 训练不足，暂时只能说明工程流程可用，还不能判断其最终性能。
5. 当前最主要的问题仍是 worst-class robust accuracy 较低，需要进一步加入 per-class 鲁棒性调节策略。

## 五、当前问题

1. 当前实现仍是 FAIR-TAT-like 轻量复现，还没有完全实现原文中的完整 per-class 调节。
2. 强攻击评测下最差类鲁棒性仍然偏低，说明训练攻击和评测攻击之间存在差距。
3. ResNet18 需要更大训练规模和更多 epoch 才能公平比较。
4. 医疗图像数据集目前处于调研阶段，尚未开始 ISIC 2019 实际训练。

## 六、下周计划

1. 尝试 per-class robust deficit 调节 target prior 或 loss weight。
2. 扩大 ResNet18 训练规模，例如 20k 或 full CIFAR-10。
3. 继续做 multi-seed 稳定性实验，观察 worst-class robust accuracy 波动。
4. 整理 ISIC 2019 数据下载、字段解析和 patient-level split 方案。
5. 准备将 CIFAR-10 的 class-wise fairness 评测迁移到医疗图像属性分组评测。

## 七、材料位置
实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：
- FAIR-TAT 对齐笔记：`week3/docs/fair_tat_alignment_notes.md`
- ISIC 2019 调研笔记：`week3/docs/isic2019_fairness_notes.md`
- 当前结果总表：`week3/docs/week3_current_results.md`