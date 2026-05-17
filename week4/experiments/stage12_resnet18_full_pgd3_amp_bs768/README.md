# stage12_resnet18_full_pgd3_amp_bs768

## 目的

测试中等大 batch size 在 AMP 加速下的效果，判断是否能在速度与鲁棒公平性之间取得比 batch size 1536 更好的平衡。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：768
- AMP：启用
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-7、targeted PGD-3
- `prior_mode=blended`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.7170
- clean worst-class accuracy：0.4280
- FGSM robust accuracy：0.3529
- FGSM worst-class robust accuracy：0.0424
- multi-step FGSM robust accuracy：0.3251
- multi-step FGSM worst-class robust accuracy：0.0233
- untargeted PGD robust accuracy：0.3207
- untargeted PGD worst-class robust accuracy：0.0212
- targeted PGD original accuracy：0.5844
- targeted PGD worst-class robust accuracy：0.3047
- targeted PGD target success rate：0.2220

## 初步结论

batch size 768 比 1536 略好，平均 untargeted PGD robust accuracy 进一步提高，但 clean accuracy 和 worst-class robust accuracy 仍明显低于 stage09。说明大 batch + AMP 更适合快速探索，不适合作为当前最终主结果配置。
