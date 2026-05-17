# stage13_resnet18_full_adv08_pgd3_bs128

## 目的

在 stage09 的稳定 full CIFAR-10 配置基础上，将 `adv_weight` 从 0.7 提高到 0.8，观察更高对抗损失权重是否能进一步改善 worst-class robust accuracy。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：128
- AMP：未启用
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-7、targeted PGD-3
- `adv_weight=0.8`
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.8004
- clean worst-class accuracy：0.6390
- FGSM robust accuracy：0.3245
- FGSM worst-class robust accuracy：0.1144
- multi-step FGSM robust accuracy：0.2732
- multi-step FGSM worst-class robust accuracy：0.0827
- untargeted PGD robust accuracy：0.2720
- untargeted PGD worst-class robust accuracy：0.0754
- targeted PGD original accuracy：0.6423
- targeted PGD worst-class robust accuracy：0.4586
- targeted PGD target success rate：0.1904

## 与 stage09 对比

| 指标 | stage09 adv0.7 | stage13 adv0.8 |
|---|---:|---:|
| clean accuracy | **0.8035** | 0.8004 |
| clean worst-class accuracy | 0.6170 | **0.6390** |
| untargeted PGD robust accuracy | 0.2476 | **0.2720** |
| untargeted PGD worst-class accuracy | 0.0706 | **0.0754** |
| targeted PGD original accuracy | 0.6338 | **0.6423** |
| targeted PGD worst-class accuracy | 0.4261 | **0.4586** |

## 初步结论

提高 `adv_weight` 到 0.8 后，clean accuracy 基本保持，同时 clean worst-class、untargeted PGD robust、untargeted PGD worst-class、targeted PGD original 和 targeted PGD worst-class 均有提升。当前 stage13 是第四周最优结果候选。
