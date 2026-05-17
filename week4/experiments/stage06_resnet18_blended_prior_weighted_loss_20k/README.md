# stage06_resnet18_blended_prior_weighted_loss_20k

## 目的

验证第四周提出的 per-class 调节策略是否能改善 ResNet18 的 worst-class robust accuracy。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100 16G
- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 20k
- 测试数据：CIFAR-10 3k
- 训练轮数：4
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-3、untargeted PGD-5、targeted PGD-3
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`
- `adv_weight=0.7`
- `prior_smoothing=10.0`

## 结果摘要

- clean accuracy：0.6417
- clean worst-class accuracy：0.4306
- FGSM robust accuracy：0.2188
- FGSM worst-class robust accuracy：0.0889
- multi-step FGSM robust accuracy：0.2725
- multi-step FGSM worst-class robust accuracy：0.1222
- untargeted PGD robust accuracy：0.2227
- untargeted PGD worst-class robust accuracy：0.0889
- targeted PGD original accuracy：0.5234
- targeted PGD worst-class robust accuracy：0.3364
- targeted PGD target success rate：0.2529

## 与 stage05 对比

| 指标 | stage05 ResNet18 baseline | stage06 ResNet18 per-class |
|---|---:|---:|
| clean accuracy | 0.6407 | 0.6417 |
| clean worst-class accuracy | 0.2052 | **0.4306** |
| untargeted PGD robust accuracy | **0.2295** | 0.2227 |
| untargeted PGD worst-class accuracy | 0.0000 | **0.0889** |
| targeted PGD original accuracy | 0.4844 | **0.5234** |
| targeted PGD worst-class accuracy | 0.1165 | **0.3364** |

## 初步结论

per-class 调节策略在 ResNet18 上效果明显：平均鲁棒性基本保持，同时显著提升了 clean、untargeted PGD 和 targeted PGD 下的 worst-class accuracy。该实验可以作为第四周最主要结果。
