# stage09_resnet18_full_blended_prior_weighted_loss

## 目的

使用完整 CIFAR-10 训练集和完整测试集验证 ResNet18 + per-class 组合调节策略是否能在大规模设置下保持鲁棒公平性提升。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100 32G
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：128
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-7、targeted PGD-3
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`
- `adv_weight=0.7`

## 结果摘要

- clean accuracy：0.8035
- clean worst-class accuracy：0.6170
- FGSM robust accuracy：0.3010
- FGSM worst-class robust accuracy：0.1144
- multi-step FGSM robust accuracy：0.2507
- multi-step FGSM worst-class robust accuracy：0.0706
- untargeted PGD robust accuracy：0.2476
- untargeted PGD worst-class robust accuracy：0.0706
- targeted PGD original accuracy：0.6338
- targeted PGD worst-class robust accuracy：0.4261
- targeted PGD target success rate：0.2173

## 初步结论

full CIFAR-10 结果明显优于 20k 子集实验，说明扩大训练数据和训练轮数后，ResNet18 + per-class 调节策略能够同时提升 clean accuracy、targeted PGD 平均鲁棒性和 worst-class robust accuracy。

该实验可以作为第四周最重要的主结果。
