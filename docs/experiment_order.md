# 实验先后顺序说明

本项目当前实验按“先跑通流程，再加入攻击，再做对抗训练，最后扩大规模对比”的顺序推进。被问到实验顺序时，可以按下面逻辑说明。

## stage01：smoke test 与攻击流程验证

目录：`experiments/stage01_smoke_test`

目的：先确认环境、CIFAR-10 数据加载、基础训练和模型保存流程可以跑通。

完成内容：

- 跑通 `SimpleCNN` 单 batch 训练。
- 验证 loss、反向传播和参数更新正常。
- 接入 FGSM attack。
- 接入 targeted PGD attack。
- 验证 targeted PGD 能推动样本向指定目标类别靠近。

这一步是所有后续实验的基础。

## stage02：5k clean baseline

目录：`experiments/stage02_cifar10_baseline_cpu_5k`

目的：在小规模 CIFAR-10 子集上训练普通模型，作为后续对抗训练的对照组。

完成内容：

- 使用 5000 个训练样本、1000 个测试样本。
- 训练 SmallCNN。
- 记录 clean accuracy、worst-class accuracy。
- 加入 FGSM 和 targeted PGD 对抗评测。

这一步用于回答：普通训练模型在 clean 和 adversarial 场景下表现如何。

## stage03：5k FGSM adversarial training

目录：`experiments/stage03_fgsm_adversarial_training_cpu_5k`

目的：验证最简单的对抗训练入口是否有效。

完成内容：

- 在训练中加入 FGSM adversarial examples。
- 对比 clean baseline 的 clean accuracy 和 FGSM adversarial accuracy。
- 观察 clean performance 与 adversarial robustness 的 trade-off。

这一步用于回答：FGSM 对抗训练是否能提升 FGSM 鲁棒性。

## stage04：5k targeted PGD adversarial training

目录：`experiments/stage04_targeted_pgd_adversarial_training_cpu_5k`

目的：初步验证 targeted adversarial training 入口，为 FAIR-TAT 复现做准备。

完成内容：

- 使用 targeted PGD 生成目标攻击样本。
- 训练时要求模型在 targeted adversarial examples 上仍预测原类别。
- target class 暂用 `(label + 1) % 10` 的简单规则。

这一步不是完整 FAIR-TAT，只是验证 targeted training 流程能跑通。

## stage05：10k clean baseline

目录：`experiments/stage05_cifar10_baseline_cpu_10k`

目的：扩大训练规模，得到更稳定的普通训练对照组。

完成内容：

- 使用 10000 个训练样本、2000 个测试样本。
- 训练 5 个 epoch。
- 记录 clean accuracy、worst-class accuracy 和对抗评测指标。

这一步作为 10k 主实验的 baseline。

## stage06：10k FGSM adversarial training

目录：`experiments/stage06_fgsm_adversarial_training_cpu_10k`

目的：在更大训练规模下验证 FGSM adversarial training 的效果。

完成内容：

- 与 stage05 使用相同数据规模和训练轮数。
- 对比 clean baseline 的 FGSM adversarial accuracy。
- 观察 FGSM 鲁棒性提升和 clean accuracy 损失。

这一步用于支撑“FGSM AT 提升 FGSM 鲁棒性，但会牺牲部分 clean 表现”的结论。

## stage07：10k targeted PGD adversarial training

目录：`experiments/stage07_targeted_pgd_adversarial_training_cpu_10k`

目的：在同规模设置下验证 targeted PGD adversarial training 的效果。

完成内容：

- 与 stage05、stage06 使用相同数据规模和训练轮数。
- 使用 targeted PGD adversarial examples 进行训练。
- 重点观察 targeted PGD original accuracy 和 target success rate。

这一步最接近 FAIR-TAT 的 targeted adversarial training 思路，但 target class 选择策略仍需后续对齐论文。

## 被问到时的简短说法

可以这样回答：

> 我先做了 smoke test，确认环境、数据加载和基础训练流程能跑通；然后在 5k 数据上分别做 clean baseline、FGSM 对抗训练和 targeted PGD 对抗训练，验证攻击和训练入口；接着做类别级分析，观察普通对抗训练是否造成类别间表现不均衡；最后把训练规模扩大到 10k，用相同设置比较 clean baseline、FGSM AT 和 targeted PGD AT，作为本周的主实验结果。

## 关键逻辑

- 先有 clean baseline，才能判断对抗训练有没有带来变化。
- 先做 FGSM AT，是因为它计算简单，适合作为对抗训练入口。
- 再做 targeted PGD AT，是因为它更接近 FAIR-TAT 的 targeted adversarial training 思路。
- 最后扩大到 10k，是为了让结果比最初的 smoke test 和 5k 小实验更稳定。
