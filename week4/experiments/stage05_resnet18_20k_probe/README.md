# stage05_resnet18_20k_probe

## 目的

在 AI Studio 的 Tesla V100 16G 上运行 ResNet18 版本 FAIR-TAT-like 训练，验证更强 backbone 在 CIFAR-10 20k 设置下的表现。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100 16G
- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 20k
- 测试数据：CIFAR-10 3k
- 训练轮数：4
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-3、untargeted PGD-5、targeted PGD-3
- `adv_weight=0.7`
- `prior_smoothing=10.0`
- `prior_mode=false_positive`

## 结果摘要

- clean accuracy：0.6407
- clean worst-class accuracy：0.2052
- FGSM robust accuracy：0.2148
- FGSM worst-class robust accuracy：0.0097
- multi-step FGSM robust accuracy：0.2949
- multi-step FGSM worst-class robust accuracy：0.0194
- untargeted PGD robust accuracy：0.2295
- untargeted PGD worst-class robust accuracy：0.0000
- targeted PGD original accuracy：0.4844
- targeted PGD worst-class robust accuracy：0.1165
- targeted PGD target success rate：0.2881

## 初步观察

ResNet18 的平均 clean accuracy 和 targeted PGD original accuracy 高于 20k SmallCNN baseline，但 worst-class robust accuracy 明显偏低，尤其 untargeted PGD worst-class 为 0。说明更强 backbone 能提高平均表现，但不能自动解决最差类鲁棒性问题。

下一步不建议直接 full ResNet18，而应优先运行 ResNet18 的 per-class 调节版本。
