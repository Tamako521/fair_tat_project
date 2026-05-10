# stage04_resnet18_5k_probe

## 目的

在中等规模 CIFAR-10 子集上验证 ResNet18 版本 FAIR-TAT-like 训练是否可运行，并观察更强模型结构的初步趋势。

## 配置

- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 子集 5000
- 测试数据：CIFAR-10 子集 1000
- 训练轮数：2
- 随机种子：42
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-3、untargeted PGD-5、targeted PGD-3
- `adv_weight=0.7`
- `prior_smoothing=10.0`

## 结果摘要

- clean accuracy：0.4470
- clean worst-class accuracy：0.0899
- FGSM robust accuracy：0.1582
- FGSM worst-class robust accuracy：0.0000
- multi-step FGSM robust accuracy：0.1895
- multi-step FGSM worst-class robust accuracy：0.0000
- untargeted PGD robust accuracy：0.1641
- untargeted PGD worst-class robust accuracy：0.0000
- targeted PGD original accuracy：0.3828
- targeted PGD worst-class robust accuracy：0.0784
- targeted PGD target success rate：0.2344

## 初步观察

ResNet18 入口已经跑通，但 5k 数据和 2 epoch 明显不足，worst-class 指标仍偏低。下一步应增加训练轮数或训练规模，再判断 ResNet18 是否优于 SmallCNN。
