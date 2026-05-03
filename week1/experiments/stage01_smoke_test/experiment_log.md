# 实验记录：基础训练与 FGSM 对抗攻击验证

## 实验目的

验证当前项目环境、CIFAR-10 数据加载、基础训练流程、FGSM 对抗样本生成流程和 targeted PGD 攻击流程是否能够跑通，为后续复现 FAIR-TAT 的 targeted adversarial training 做准备。

## 实验设置

- 数据集：CIFAR-10
- 模型：SimpleCNN
- batch size：32
- 优化器：Adam
- 学习率：0.001
- 攻击方法：FGSM、targeted PGD
- 攻击类型：untargeted attack、targeted attack
- 扰动强度：epsilon = 8 / 255
- PGD 步长：alpha = 2 / 255
- PGD 迭代步数：5
- targeted PGD 目标类别构造：`target_label = (label + 1) % 10`
- 运行设备：CPU
- 训练规模：单 batch smoke test

## 运行输出

```text
device: cpu
clean loss: 2.3192856311798096
fgsm adversarial loss: 2.343320846557617
clean target loss: 2.8706908226013184
targeted pgd loss to target: 2.568723678588867
targeted pgd loss to original: 2.099764108657837
训练成功！
FGSM 对抗攻击流程验证成功！
targeted PGD 对抗攻击流程验证成功！
模型已保存
```

## 结果说明

本次实验不是正式性能实验，而是用于验证代码流程。当前结果说明：

1. CIFAR-10 数据加载可以正常运行。
2. 模型能够完成前向传播、损失计算、反向传播和参数更新。
3. 已初步接入 FGSM 对抗样本生成流程。
4. FGSM 对抗样本上的 loss 高于 clean loss，说明扰动对模型输出产生了影响。
5. 已初步接入 targeted PGD 攻击流程，能够按照指定目标类别生成对抗样本。
6. targeted PGD 后目标类别损失从 `2.8707` 降到 `2.5687`，说明攻击方向已能推动样本接近目标类别。
7. 后续可以在此基础上继续整理独立攻击模块，并进一步对齐 FAIR-TAT 的 targeted adversarial training。

## 周报可用表述

在基础训练流程跑通后，我进一步接入了最小 FGSM 与 targeted PGD 对抗攻击验证流程。当前脚本可以在 CIFAR-10 的单 batch 上完成普通训练，并生成 untargeted FGSM 对抗样本和指定目标类别的 targeted PGD 对抗样本，输出 clean loss、adversarial loss 与 target loss。该实验主要用于验证对抗样本生成流程是否能够接入训练代码，不作为正式性能结果。后续将基于该流程整理独立攻击模块，并进一步对齐 FAIR-TAT 的 targeted adversarial training。

## 后续计划

1. 将 FGSM 与 targeted PGD 攻击函数进一步整理为独立攻击模块。
2. 实现 untargeted PGD，用多步迭代扰动替代单步 FGSM。
3. 继续阅读 FAIR-TAT，明确论文中 target class 的选择策略。
4. 增加 clean accuracy、adversarial accuracy 和 worst-class accuracy 评测。
