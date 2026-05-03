# CIFAR-10 小型 baseline 与对抗评测实验

## 实验目的

在 CPU 环境下完成一个小型但完整的训练与评测闭环，验证后续 FAIR-TAT 复现所需的基础组件：

1. CIFAR-10 基础分类训练。
2. clean accuracy 评测。
3. worst-class accuracy 统计。
4. FGSM 对抗评测。
5. targeted PGD 目标攻击评测。

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
- FGSM epsilon：8 / 255
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5
- 设备：CPU

## 主要结果

```text
epoch 1/3 | train loss: 2.0733 | train acc: 0.2398 | clean test acc: 0.3460 | worst-class acc: 0.0000
epoch 2/3 | train loss: 1.6857 | train acc: 0.3880 | clean test acc: 0.4030 | worst-class acc: 0.1165
epoch 3/3 | train loss: 1.5287 | train acc: 0.4480 | clean test acc: 0.4480 | worst-class acc: 0.2727
最终 clean accuracy: 0.4480
最终 worst-class accuracy: 0.2727
FGSM adversarial accuracy: 0.0781
targeted PGD original accuracy: 0.2539
targeted PGD target success rate: 0.3496
```

## 结果说明

本实验不是正式性能对比，而是小规模前置验证。结果说明：

- 基础训练流程已经可以完成多 epoch 训练。
- clean accuracy 随训练轮数提升，说明模型正在学习 CIFAR-10 分类特征。
- FGSM adversarial accuracy 明显低于 clean accuracy，说明当前基础模型对对抗扰动较敏感。
- targeted PGD target success rate 达到 `0.3496`，说明目标攻击流程可以推动模型输出向指定目标类别变化。
- worst-class accuracy 已纳入记录，后续可以继续扩展为 FAIR-TAT 所关注的类别级公平性评测。

## 文件说明

- `train_cifar10_baseline.py`：本次实验使用的训练脚本快照。
- `train_log.txt`：训练与评测日志。
- `metrics.json`：结构化实验指标。
- `model.pth`：本次实验保存的模型权重。

