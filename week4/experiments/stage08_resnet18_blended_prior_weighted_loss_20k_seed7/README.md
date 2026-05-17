# stage08_resnet18_blended_prior_weighted_loss_20k_seed7

## 目的

换随机种子复跑 ResNet18 + per-class 组合策略，检查 stage06 的提升是否稳定。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：CIFAR-10 20k
- 测试数据：CIFAR-10 3k
- 训练轮数：4
- 随机种子：7
- 训练攻击：targeted PGD-3
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.6867
- clean worst-class accuracy：0.3084
- FGSM robust accuracy：0.2432
- FGSM worst-class robust accuracy：0.0187
- multi-step FGSM robust accuracy：0.3008
- multi-step FGSM worst-class robust accuracy：0.0187
- untargeted PGD robust accuracy：0.2549
- untargeted PGD worst-class robust accuracy：0.0187
- targeted PGD original accuracy：0.5547
- targeted PGD worst-class robust accuracy：0.2430
- targeted PGD target success rate：0.2207

## 初步观察

换 seed 后平均指标继续提升，但 worst-class robust accuracy 低于 stage06，说明 per-class 调节方向有效，但在 20k 子集和较小评测集下仍存在波动。下一步应使用 full CIFAR-10、完整测试集和更强评测来确认结论。
