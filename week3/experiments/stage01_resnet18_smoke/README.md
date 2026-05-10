# stage01_resnet18_smoke

## 目的

验证第三周新增的 ResNet18 模型入口、multi-step FGSM 评测和 untargeted PGD 评测可以正常跑通。

## 配置

- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 子集 128
- 测试数据：CIFAR-10 子集 64
- 训练轮数：1
- 训练攻击：targeted PGD-2
- 评测攻击：FGSM、multi-step FGSM-2、untargeted PGD-2、targeted PGD-2
- `adv_weight=0.7`
- `prior_smoothing=10.0`

## 结果摘要

- clean accuracy：0.2188
- FGSM robust accuracy：0.0000
- multi-step FGSM robust accuracy：0.0000
- untargeted PGD robust accuracy：0.0000
- targeted PGD original accuracy：0.1250
- targeted PGD target success rate：0.5000

该实验只用于验证流程，不用于与第二周正式结果比较。
