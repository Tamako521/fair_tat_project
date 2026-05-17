# stage07_resnet18_class_weighted_loss_20k

## 目的

验证 ResNet18 上仅使用 class-weighted adversarial loss 是否能够改善 worst-class robust accuracy，并与 stage06 的组合策略对比。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100 16G
- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 20k
- 测试数据：CIFAR-10 3k
- 训练轮数：4
- 训练攻击：targeted PGD-3
- `prior_mode=false_positive`
- `loss_weight_mode=robust_deficit`
- `deficit_source=targeted_pgd`

## 结果摘要

- clean accuracy：0.6037
- clean worst-class accuracy：0.3132
- FGSM robust accuracy：0.1934
- FGSM worst-class robust accuracy：0.0097
- multi-step FGSM robust accuracy：0.2324
- multi-step FGSM worst-class robust accuracy：0.0549
- untargeted PGD robust accuracy：0.1992
- untargeted PGD worst-class robust accuracy：0.0194
- targeted PGD original accuracy：0.4570
- targeted PGD worst-class robust accuracy：0.2857
- targeted PGD target success rate：0.3252

## 与 stage06 对比

| 指标 | stage06 组合策略 | stage07 仅 loss weight |
|---|---:|---:|
| clean accuracy | **0.6417** | 0.6037 |
| clean worst-class accuracy | **0.4306** | 0.3132 |
| untargeted PGD robust accuracy | **0.2227** | 0.1992 |
| untargeted PGD worst-class accuracy | **0.0889** | 0.0194 |
| targeted PGD original accuracy | **0.5234** | 0.4570 |
| targeted PGD worst-class accuracy | **0.3364** | 0.2857 |

## 初步结论

class-weighted adversarial loss 单独使用时可以提升 targeted PGD worst-class robust accuracy，但对 untargeted PGD worst-class 的改善有限。stage06 的组合策略明显更强，说明 blended target prior 与 class-weighted loss 具有互补作用。
