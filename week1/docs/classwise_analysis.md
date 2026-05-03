# 类别级对比分析：clean baseline 与 FGSM 对抗训练

## 分析目的

本分析用于比较 clean baseline 与 FGSM adversarial training 在 CIFAR-10 各类别上的 clean accuracy 表现差异，观察简单对抗训练是否会造成类别间性能变化不均衡，从而为后续 FAIR-TAT 复现中的公平性分析提供前置依据。

## 实验来源

- clean baseline：`experiments/stage02_cifar10_baseline_cpu_5k/metrics.json`
- FGSM adversarial training：`experiments/stage03_fgsm_adversarial_training_cpu_5k/metrics.json`

两组实验均使用：

- 数据集：CIFAR-10
- 训练样本数：5000
- 测试样本数：1000
- 训练轮数：3
- 模型：SmallCNN
- 设备：CPU

## 总体指标对比

| 指标 | clean baseline | FGSM adversarial training | 变化 |
| --- | ---: | ---: | ---: |
| clean accuracy | 0.4480 | 0.4060 | -0.0420 |
| worst-class accuracy | 0.2727 | 0.0112 | -0.2615 |
| FGSM adversarial accuracy | 0.0781 | 0.2148 | +0.1367 |
| targeted PGD original accuracy | 0.2539 | 0.2949 | +0.0410 |
| targeted PGD target success rate | 0.3496 | 0.1934 | -0.1562 |

## 类别级 clean accuracy 对比

| CIFAR-10 类别编号 | clean baseline | FGSM adversarial training | 变化 |
| ---: | ---: | ---: | ---: |
| 0 | 0.5349 | 0.4302 | -0.1047 |
| 1 | 0.4000 | 0.4762 | +0.0762 |
| 2 | 0.2921 | 0.0112 | -0.2809 |
| 3 | 0.4272 | 0.1165 | -0.3107 |
| 4 | 0.2766 | 0.1170 | -0.1596 |
| 5 | 0.5278 | 0.4444 | -0.0833 |
| 6 | 0.5567 | 0.7320 | +0.1753 |
| 7 | 0.2727 | 0.5152 | +0.2424 |
| 8 | 0.7980 | 0.6364 | -0.1616 |
| 9 | 0.3917 | 0.5167 | +0.1250 |

## 主要观察

1. FGSM adversarial training 提升了整体 FGSM adversarial accuracy，从 `0.0781` 提升到 `0.2148`。
2. clean accuracy 从 `0.4480` 下降到 `0.4060`，说明简单对抗训练带来了一定 clean performance 损失。
3. worst-class accuracy 从 `0.2727` 明显下降到 `0.0112`，说明类别级最差表现恶化。
4. 类别 6、7、9 在 FGSM adversarial training 后 clean accuracy 上升，其中类别 7 从 `0.2727` 提升到 `0.5152`。
5. 类别 2、3、4 的 clean accuracy 明显下降，类别 2 下降到 `0.0112`，成为新的最差类别。
6. 这说明简单 FGSM adversarial training 的收益并没有均匀分布到所有类别，可能会强化部分类别、削弱另一些类别。

## 与 FAIR-TAT 的关系

FAIR-TAT 关注的问题是：传统对抗训练可能提升模型整体鲁棒性，但不同类别之间的鲁棒性和准确率变化并不均衡。本次小规模实验也观察到了类似现象：

- 对抗训练后 FGSM adversarial accuracy 提升。
- 但 clean accuracy 和 worst-class accuracy 下降。
- 不同类别的变化方向不同，有些类别提升，有些类别明显下降。

因此，后续复现 FAIR-TAT 时，需要重点关注 target class 选择策略和 targeted adversarial training 是否能够缓解这种类别间表现不均衡的问题。

## 周报可用表述

在 clean baseline 与 FGSM adversarial training 的对比中，我进一步分析了 CIFAR-10 各类别的 clean accuracy 变化。结果显示，FGSM 对抗训练虽然将 FGSM adversarial accuracy 从 `0.0781` 提升到 `0.2148`，但 worst-class accuracy 从 `0.2727` 下降到 `0.0112`，并且不同类别的表现变化差异较大。例如类别 6、7、9 的准确率有所提升，而类别 2、3、4 的准确率明显下降。这说明简单对抗训练可能带来类别级表现不均衡的问题，也进一步说明后续需要复现 FAIR-TAT 中的 targeted adversarial training 方法来改善最差类别表现和类别间公平性。

## 后续计划

1. 在现有评测代码中加入 adversarial class accuracy，比较各类别在 FGSM 和 targeted PGD 攻击下的鲁棒性差异。
2. 尝试 targeted PGD adversarial training，并与 clean baseline、FGSM adversarial training 进行对比。
3. 阅读 FAIR-TAT 方法部分，明确论文中的 target class 选择策略，而不是继续使用简单的 `(label + 1) % 10`。
4. 将类别级分析扩展到 worst-class accuracy、class-wise robust accuracy 和类别间差异度量。

