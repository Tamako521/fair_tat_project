# 10k 训练规模实验对比

## 实验目的

在前置 smoke test 和 5k 小规模实验基础上，进一步扩大训练数据量，对 clean baseline、FGSM adversarial training 和 targeted PGD adversarial training 进行同规模对比，观察不同训练方式在 clean accuracy、adversarial accuracy 和 worst-class accuracy 上的差异。

## 统一实验设置

- 数据集：CIFAR-10
- 训练样本数：10000
- 测试样本数：2000
- 对抗评测样本数：1000
- 模型：SmallCNN
- 训练轮数：5
- batch size：64
- 优化器：Adam
- 学习率：0.001
- FGSM epsilon：8 / 255
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5
- 设备：CPU

## 总体结果对比

| 实验 | clean accuracy | worst-class accuracy | FGSM adversarial accuracy | targeted PGD original accuracy | targeted PGD target success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean baseline | 0.5750 | 0.3333 | 0.0684 | 0.2051 | 0.5918 |
| FGSM adversarial training | 0.5120 | 0.2646 | 0.2246 | 0.3945 | 0.2119 |
| targeted PGD adversarial training | 0.4870 | 0.2629 | 0.1201 | 0.4756 | 0.2998 |

## 训练过程摘要

### clean baseline

```text
epoch 1/5 | train loss: 1.8779 | train acc: 0.3139 | clean test acc: 0.4205 | worst-class acc: 0.0952
epoch 2/5 | train loss: 1.4661 | train acc: 0.4717 | clean test acc: 0.5035 | worst-class acc: 0.2328
epoch 3/5 | train loss: 1.3199 | train acc: 0.5318 | clean test acc: 0.5215 | worst-class acc: 0.2963
epoch 4/5 | train loss: 1.2028 | train acc: 0.5731 | clean test acc: 0.5560 | worst-class acc: 0.3545
epoch 5/5 | train loss: 1.1269 | train acc: 0.6002 | clean test acc: 0.5750 | worst-class acc: 0.3333
```

### FGSM adversarial training

```text
epoch 1/5 | train loss: 2.1034 | train acc: 0.2850 | clean test acc: 0.3860 | worst-class acc: 0.1111
epoch 2/5 | train loss: 1.8670 | train acc: 0.4158 | clean test acc: 0.4345 | worst-class acc: 0.0577
epoch 3/5 | train loss: 1.7734 | train acc: 0.4705 | clean test acc: 0.4670 | worst-class acc: 0.1443
epoch 4/5 | train loss: 1.7162 | train acc: 0.4987 | clean test acc: 0.4730 | worst-class acc: 0.0952
epoch 5/5 | train loss: 1.6795 | train acc: 0.5206 | clean test acc: 0.5120 | worst-class acc: 0.2646
```

### targeted PGD adversarial training

```text
epoch 1/5 | train loss: 1.8764 | train acc: 0.2849 | clean test acc: 0.3510 | worst-class acc: 0.0302
epoch 2/5 | train loss: 1.5242 | train acc: 0.4030 | clean test acc: 0.4280 | worst-class acc: 0.2067
epoch 3/5 | train loss: 1.4446 | train acc: 0.4469 | clean test acc: 0.4315 | worst-class acc: 0.1289
epoch 4/5 | train loss: 1.3757 | train acc: 0.4787 | clean test acc: 0.4630 | worst-class acc: 0.1759
epoch 5/5 | train loss: 1.3292 | train acc: 0.5101 | clean test acc: 0.4870 | worst-class acc: 0.2629
```

## 主要观察

1. clean baseline 的 clean accuracy 最高，达到 `0.5750`，但 FGSM adversarial accuracy 只有 `0.0684`，说明普通训练模型对 FGSM 扰动较脆弱。
2. FGSM adversarial training 将 FGSM adversarial accuracy 从 `0.0684` 提升到 `0.2246`，但 clean accuracy 从 `0.5750` 降到 `0.5120`。
3. targeted PGD adversarial training 的 targeted PGD original accuracy 达到 `0.4756`，是三组中最高，说明它对当前 targeted PGD 攻击更有针对性。
4. targeted PGD adversarial training 的 clean accuracy 为 `0.4870`，低于 clean baseline 和 FGSM adversarial training，说明 targeted adversarial training 也存在 clean performance 损失。
5. FGSM adversarial training 和 targeted PGD adversarial training 的 worst-class accuracy 都低于 clean baseline，说明对抗训练后的类别级最差表现仍需进一步优化。

## 与 FAIR-TAT 的关系

这组三方法同规模实验进一步说明：

- 普通训练模型 clean accuracy 较高，但对抗鲁棒性较弱。
- FGSM adversarial training 能明显提升 FGSM 鲁棒性。
- targeted PGD adversarial training 更能提升 targeted PGD 场景下的原类别保持率。
- 两类对抗训练都会带来不同程度的 clean accuracy 和 worst-class accuracy 损失。

这些现象与 FAIR-TAT 的研究动机一致：对抗训练不能只关注整体鲁棒性，还需要关注不同类别之间的鲁棒性差异和最差类别表现。

## 周报可用表述

在初步 5k 小规模实验之后，我进一步将训练规模扩大到 10000 个 CIFAR-10 训练样本、2000 个测试样本和 5 个 epoch，并比较了 clean baseline、FGSM adversarial training 和 targeted PGD adversarial training。结果显示，clean baseline 的 clean accuracy 最高，为 `0.5750`，但 FGSM adversarial accuracy 只有 `0.0684`；FGSM adversarial training 将 FGSM adversarial accuracy 提升到 `0.2246`；targeted PGD adversarial training 则将 targeted PGD original accuracy 提升到 `0.4756`。这说明不同训练方式对应不同类型的鲁棒性提升，但也都会带来一定 clean accuracy 和 worst-class accuracy 损失，后续需要结合 FAIR-TAT 的 targeted adversarial training 策略进一步改善类别级鲁棒公平性。

