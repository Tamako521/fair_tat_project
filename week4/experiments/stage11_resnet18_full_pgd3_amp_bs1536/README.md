# stage11_resnet18_full_pgd3_amp_bs1536

## 目的

作为 stage09 的控制变量实验：保持 PGD-3 算法配置不变，只启用 AMP、多 worker 和超大 batch，判断 stage10 变差主要来自 PGD-5，还是超大 batch / AMP 本身也影响结果。

## 配置

- 平台：百度 AI Studio
- 设备：Tesla V100
- 模型：ResNet18 for CIFAR-10
- 训练数据：full CIFAR-10 train
- 测试数据：full CIFAR-10 test
- 训练轮数：8
- batch size：1536
- AMP：启用
- 训练攻击：targeted PGD-3
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-7、targeted PGD-3
- `prior_mode=blended`
- `loss_weight_mode=robust_deficit`

## 结果摘要

- clean accuracy：0.7033
- clean worst-class accuracy：0.3890
- FGSM robust accuracy：0.3422
- FGSM worst-class robust accuracy：0.0487
- multi-step FGSM robust accuracy：0.3006
- multi-step FGSM worst-class robust accuracy：0.0288
- untargeted PGD robust accuracy：0.2947
- untargeted PGD worst-class robust accuracy：0.0221
- targeted PGD original accuracy：0.5766
- targeted PGD worst-class robust accuracy：0.2780
- targeted PGD target success rate：0.2248

## 初步结论

与 stage09 相比，超大 batch + AMP 提高了部分平均鲁棒性指标，但明显降低 clean accuracy 和 worst-class robust accuracy。说明 stage10 的下降不只来自 PGD-5，超大 batch 也可能影响模型泛化和类别均衡性。

后续若继续追求速度，建议测试中等 batch size，例如 512 或 768，而不是继续使用 1536。
