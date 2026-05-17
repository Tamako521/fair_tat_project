# stage14_resnet18_full_adv08_pgd3_bs128_seed7

## 目的

换随机种子复跑 stage13，验证 `adv_weight=0.8` 的 full CIFAR-10 最优配置是否稳定。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：128
- 随机种子：7
- AMP：未启用
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-7、targeted PGD-3
- `adv_weight=0.8`
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.8012
- clean worst-class accuracy：0.6160
- FGSM robust accuracy：0.3252
- FGSM worst-class robust accuracy：0.1293
- multi-step FGSM robust accuracy：0.2788
- multi-step FGSM worst-class robust accuracy：0.0827
- untargeted PGD robust accuracy：0.2747
- untargeted PGD worst-class robust accuracy：0.0827
- targeted PGD original accuracy：0.6580
- targeted PGD worst-class robust accuracy：0.4687
- targeted PGD target success rate：0.1970

## 与 stage13 对比

| 指标 | stage13 seed42 | stage14 seed7 |
|---|---:|---:|
| clean accuracy | 0.8004 | **0.8012** |
| clean worst-class accuracy | **0.6390** | 0.6160 |
| untargeted PGD robust accuracy | 0.2720 | **0.2747** |
| untargeted PGD worst-class accuracy | 0.0754 | **0.0827** |
| targeted PGD original accuracy | 0.6423 | **0.6580** |
| targeted PGD worst-class accuracy | 0.4586 | **0.4687** |

## 初步结论

stage14 证明 `adv_weight=0.8` 的 full CIFAR-10 配置在换 seed 后依然稳定，并且在 targeted PGD original、targeted PGD worst-class、untargeted PGD worst-class 等关键指标上进一步提升。当前可将 stage13/stage14 作为第四周最强实验组合。
