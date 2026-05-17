# stage10_resnet18_full_blended_prior_weighted_loss_pgd5_amp

## 目的

在 full CIFAR-10 上将训练攻击从 targeted PGD-3 提升到 targeted PGD-5，并启用 AMP、大 batch 和多 worker，观察更强训练攻击是否继续提升鲁棒性。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：1536
- AMP：启用
- 训练攻击：targeted PGD-5
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-10、targeted PGD-5
- `prior_mode=blended`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.6965
- clean worst-class accuracy：0.3820
- FGSM robust accuracy：0.3585
- FGSM worst-class robust accuracy：0.0530
- multi-step FGSM robust accuracy：0.3166
- multi-step FGSM worst-class robust accuracy：0.0254
- untargeted PGD robust accuracy：0.2917
- untargeted PGD worst-class robust accuracy：0.0169
- targeted PGD original accuracy：0.5723
- targeted PGD worst-class robust accuracy：0.2324
- targeted PGD target success rate：0.2500

## 与 stage09 对比

| 指标 | stage09 PGD-3 | stage10 PGD-5 AMP |
|---|---:|---:|
| clean accuracy | **0.8035** | 0.6965 |
| clean worst-class accuracy | **0.6170** | 0.3820 |
| untargeted PGD robust accuracy | 0.2476 | **0.2917** |
| untargeted PGD worst-class accuracy | **0.0706** | 0.0169 |
| targeted PGD original accuracy | **0.6338** | 0.5723 |
| targeted PGD worst-class accuracy | **0.4261** | 0.2324 |

## 初步结论

更强训练攻击提高了平均 untargeted PGD robust accuracy，但明显牺牲 clean accuracy 和 worst-class robust accuracy。当前不建议继续盲目增加 PGD steps，stage09 仍然是更均衡的主结果。
