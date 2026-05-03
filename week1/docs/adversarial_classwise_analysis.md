# 对抗类别级鲁棒性分析

## 分析目的

在已有 clean baseline 与 FGSM adversarial training 的基础上，进一步分析各类别在 FGSM 和 targeted PGD 攻击下的表现差异。该分析比 clean class accuracy 更接近 FAIR-TAT 关注的 class-wise robust accuracy 问题。

## 实验来源

- clean baseline 模型：`experiments/stage02_cifar10_baseline_cpu_5k/model.pth`
- FGSM adversarial training 模型：`experiments/stage03_fgsm_adversarial_training_cpu_5k/model.pth`
- 评测脚本：`scripts/analyze_adversarial_classwise.py`
- 评测样本数：512
- FGSM epsilon：8 / 255
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5

## FGSM 类别级对抗准确率

| CIFAR-10 类别编号 | clean baseline | FGSM adversarial training | 变化 |
| ---: | ---: | ---: | ---: |
| 0 | 0.1667 | 0.3333 | +0.1667 |
| 1 | 0.0328 | 0.2295 | +0.1967 |
| 2 | 0.0000 | 0.0000 | +0.0000 |
| 3 | 0.0196 | 0.0000 | -0.0196 |
| 4 | 0.0000 | 0.0000 | +0.0000 |
| 5 | 0.1132 | 0.2642 | +0.1509 |
| 6 | 0.0980 | 0.4510 | +0.3529 |
| 7 | 0.0204 | 0.1633 | +0.1429 |
| 8 | 0.2917 | 0.3750 | +0.0833 |
| 9 | 0.0577 | 0.3269 | +0.2692 |

## targeted PGD 原类别保持率

该指标表示 targeted PGD 攻击后，模型仍然预测为原始类别的比例，数值越高说明模型越不容易被目标攻击推离原类别。

| CIFAR-10 类别编号 | clean baseline | FGSM adversarial training | 变化 |
| ---: | ---: | ---: | ---: |
| 0 | 0.6458 | 0.3958 | -0.2500 |
| 1 | 0.1311 | 0.3607 | +0.2295 |
| 2 | 0.0588 | 0.0000 | -0.0588 |
| 3 | 0.2353 | 0.0196 | -0.2157 |
| 4 | 0.0000 | 0.0000 | +0.0000 |
| 5 | 0.3396 | 0.3585 | +0.0189 |
| 6 | 0.2157 | 0.4510 | +0.2353 |
| 7 | 0.0816 | 0.3061 | +0.2245 |
| 8 | 0.7083 | 0.5417 | -0.1667 |
| 9 | 0.1731 | 0.5000 | +0.3269 |

## targeted PGD 目标攻击成功率

该指标表示 targeted PGD 攻击后，模型被推到指定目标类别的比例，数值越低说明目标攻击越不容易成功。

| CIFAR-10 类别编号 | clean baseline | FGSM adversarial training | 变化 |
| ---: | ---: | ---: | ---: |
| 0 | 0.1875 | 0.1042 | -0.0833 |
| 1 | 0.0984 | 0.0000 | -0.0984 |
| 2 | 0.5098 | 0.0784 | -0.4314 |
| 3 | 0.3529 | 0.0980 | -0.2549 |
| 4 | 0.5417 | 0.2708 | -0.2708 |
| 5 | 0.4906 | 0.5283 | +0.0377 |
| 6 | 0.2157 | 0.3137 | +0.0980 |
| 7 | 0.5306 | 0.2245 | -0.3061 |
| 8 | 0.1250 | 0.2708 | +0.1458 |
| 9 | 0.4808 | 0.0769 | -0.4038 |

## 主要观察

1. FGSM adversarial training 显著提升了多数类别的 FGSM 对抗准确率，尤其是类别 6、9、1。
2. 类别 2 和类别 4 在 FGSM 攻击下仍然为 `0.0000`，说明它们在当前小模型和小样本设置下仍是明显脆弱类别。
3. targeted PGD 原类别保持率在类别 1、6、7、9 上提升明显，但类别 0、2、3、8 下降。
4. targeted PGD 目标攻击成功率在多数类别上下降，说明 FGSM adversarial training 对目标攻击也有一定间接防御效果。
5. 类别 5、6、8 的 targeted PGD 目标攻击成功率反而上升，说明简单 FGSM 对抗训练并不能均匀提升所有类别的目标攻击鲁棒性。

## 与 FAIR-TAT 的关系

这组结果进一步说明，简单对抗训练的收益具有明显类别差异：

- 它提升了整体 FGSM 鲁棒性。
- 它降低了多数类别的 targeted PGD 攻击成功率。
- 但仍存在部分类别几乎没有获得鲁棒性提升，甚至在 targeted PGD 场景下更容易被攻击成功。

因此，后续复现 FAIR-TAT 时，需要重点关注 targeted adversarial training 如何选择目标类别，以及这种选择是否能够改善 class-wise robust accuracy 和最差类别鲁棒性。

## 周报可用表述

在已有 clean baseline 与 FGSM adversarial training 的基础上，我进一步补充了对抗场景下的类别级鲁棒性分析。结果显示，FGSM 对抗训练可以提升多数类别的 FGSM adversarial accuracy，例如类别 6 从 `0.0980` 提升到 `0.4510`，类别 9 从 `0.0577` 提升到 `0.3269`。但类别 2 和类别 4 在 FGSM 攻击下仍然为 `0.0000`，说明简单对抗训练没有均匀改善所有类别的鲁棒性。在 targeted PGD 场景下，多数类别的攻击成功率下降，但类别 5、6、8 的目标攻击成功率反而上升。这进一步说明鲁棒性提升存在类别间不均衡问题，为后续复现 FAIR-TAT 的 targeted adversarial training 提供了实验动机。

