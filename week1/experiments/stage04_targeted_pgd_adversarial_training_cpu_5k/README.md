# CIFAR-10 targeted PGD 对抗训练实验

## 实验目的

在已有 clean baseline 与 FGSM adversarial training 的基础上，进一步尝试 targeted PGD adversarial training 的最小训练入口。该实验用于贴近 FAIR-TAT 中 targeted adversarial training 的思想，但当前 target class 暂时使用简单规则 `(label + 1) % 10`，尚未对齐论文中的目标类别选择策略。

## 实验设置

- 数据集：CIFAR-10
- 训练样本数：3000
- 测试样本数：1000
- 对抗评测样本数：512
- 模型：SmallCNN
- 训练轮数：2
- batch size：64
- 优化器：Adam
- 学习率：0.001
- 训练模式：targeted PGD adversarial training
- clean loss 权重：0.5
- adversarial loss 权重：0.5
- target class：`target_label = (label + 1) % 10`
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5
- 设备：CPU

## 主要结果

```text
epoch 1/2 | train loss: 2.1830 | train acc: 0.1717 | clean test acc: 0.2500 | worst-class acc: 0.0000 | train clean loss: 2.1972 | train adv loss: 2.1689
epoch 2/2 | train loss: 1.8373 | train acc: 0.2937 | clean test acc: 0.3580 | worst-class acc: 0.1798 | train clean loss: 1.9219 | train adv loss: 1.7528
最终 clean accuracy: 0.3580
最终 worst-class accuracy: 0.1798
FGSM adversarial accuracy: 0.0898
targeted PGD original accuracy: 0.3691
targeted PGD target success rate: 0.2852
```

## 与已有实验对比

| 指标 | clean baseline | FGSM adversarial training | targeted PGD adversarial training |
| --- | ---: | ---: | ---: |
| clean accuracy | 0.4480 | 0.4060 | 0.3580 |
| worst-class accuracy | 0.2727 | 0.0112 | 0.1798 |
| FGSM adversarial accuracy | 0.0781 | 0.2148 | 0.0898 |
| targeted PGD original accuracy | 0.2539 | 0.2949 | 0.3691 |
| targeted PGD target success rate | 0.3496 | 0.1934 | 0.2852 |

## 结果说明

本实验是 CPU 小规模前置验证，不作为最终性能结论。由于 targeted PGD adversarial training 的计算成本较高，本次只使用 3000 个训练样本和 2 个 epoch，因此与前两组实验并非严格公平对比。

当前结果显示：

- targeted PGD adversarial training 的 targeted PGD original accuracy 达到 `0.3691`，高于 clean baseline 与 FGSM adversarial training，说明该训练方式对当前 targeted PGD 攻击有一定防御效果。
- clean accuracy 为 `0.3580`，低于前两组实验，说明在小样本和短训练设置下 clean performance 受到影响。
- worst-class accuracy 为 `0.1798`，高于 FGSM adversarial training 的 `0.0112`，但低于 clean baseline 的 `0.2727`。
- 该实验已经初步跑通 targeted adversarial training 入口，后续需要进一步对齐 FAIR-TAT 的 target class 选择策略，并在相同训练设置下进行更公平的对比。

## 文件说明

- `train_cifar10_baseline.py`：本次实验使用的训练脚本快照。
- `train_log.txt`：训练与评测日志。
- `metrics.json`：结构化实验指标。
- `model.pth`：本次实验保存的模型权重。

