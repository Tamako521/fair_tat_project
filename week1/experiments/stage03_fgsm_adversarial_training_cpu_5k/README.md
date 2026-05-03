# CIFAR-10 FGSM 对抗训练实验

## 实验目的

在 clean baseline 的基础上，加入最小版 FGSM adversarial training，验证对抗训练入口是否能够跑通，并观察 clean accuracy 与 FGSM adversarial accuracy 之间的变化。

## 实验设置

- 数据集：CIFAR-10
- 训练样本数：5000
- 测试样本数：1000
- 对抗评测样本数：512
- 模型：SmallCNN
- 训练轮数：3
- batch size：64
- 优化器：Adam
- 学习率：0.001
- 训练模式：FGSM adversarial training
- clean loss 权重：0.5
- adversarial loss 权重：0.5
- FGSM epsilon：8 / 255
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5
- 设备：CPU

## 主要结果

```text
epoch 1/3 | train loss: 2.2068 | train acc: 0.2048 | clean test acc: 0.2530 | worst-class acc: 0.0000 | train clean loss: 2.1573 | train adv loss: 2.2564
epoch 2/3 | train loss: 2.0352 | train acc: 0.3272 | clean test acc: 0.3560 | worst-class acc: 0.0319 | train clean loss: 1.9036 | train adv loss: 2.1668
epoch 3/3 | train loss: 1.9116 | train acc: 0.3956 | clean test acc: 0.4060 | worst-class acc: 0.0112 | train clean loss: 1.7141 | train adv loss: 2.1091
最终 clean accuracy: 0.4060
最终 worst-class accuracy: 0.0112
FGSM adversarial accuracy: 0.2148
targeted PGD original accuracy: 0.2949
targeted PGD target success rate: 0.1934
```

## 与 clean baseline 对比

| 指标 | clean baseline | FGSM adversarial training |
| --- | ---: | ---: |
| clean accuracy | 0.4480 | 0.4060 |
| worst-class accuracy | 0.2727 | 0.0112 |
| FGSM adversarial accuracy | 0.0781 | 0.2148 |
| targeted PGD original accuracy | 0.2539 | 0.2949 |
| targeted PGD target success rate | 0.3496 | 0.1934 |

## 结果说明

本实验是 CPU 小规模前置验证，不作为正式性能结论。当前结果显示：

- FGSM adversarial training 将 FGSM adversarial accuracy 从 `0.0781` 提升到 `0.2148`，说明对抗训练确实提高了模型对 FGSM 扰动的抵抗能力。
- clean accuracy 从 `0.4480` 下降到 `0.4060`，体现了 clean performance 与 adversarial robustness 之间的初步 trade-off。
- targeted PGD target success rate 从 `0.3496` 降到 `0.1934`，说明经过 FGSM 对抗训练后，模型被推动到指定目标类别的难度有所增加。
- worst-class accuracy 明显下降，说明简单 FGSM 对抗训练可能会造成类别间性能不均衡，后续需要结合 FAIR-TAT 的 targeted adversarial training 思路继续改进。

## 文件说明

- `train_cifar10_baseline.py`：本次实验使用的训练脚本快照。
- `train_log.txt`：训练与评测日志。
- `metrics.json`：结构化实验指标。
- `model.pth`：本次实验保存的模型权重。

