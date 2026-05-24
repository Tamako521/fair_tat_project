# stage01_cifar100_resnet18_safe

## 实验目的

验证 CIFAR-100 数据集在 AI Studio GPU 环境中能否稳定运行 ResNet18 + FAIR-TAT-like targeted adversarial training。

本阶段重点是流程稳定性，不追求最终精度。

## 配置摘要

- dataset: CIFAR-100
- model: ResNet18
- train size: 5000
- test size: 1000
- epochs: 2
- batch size: 128
- PGD train steps: 2
- eval PGD steps: 3
- adv weight: 0.8
- prior mode: blended
- loss weight mode: robust_deficit
- AMP: off
- cuDNN: disabled

## 关键结果

| 指标 | 数值 |
|---|---:|
| clean accuracy | 0.0980 |
| clean worst-class accuracy | 0.0000 |
| FGSM robust accuracy | 0.0293 |
| multi-step FGSM robust accuracy | 0.0664 |
| untargeted PGD robust accuracy | 0.0547 |
| untargeted PGD worst-class robust accuracy | 0.0000 |
| targeted PGD original accuracy | 0.1055 |
| targeted PGD worst-class robust accuracy | 0.0000 |
| targeted PGD target success rate | 0.0371 |

## 结论

实验成功跑通，说明当前代码已经支持 CIFAR-100，并且 FAIR-TAT-like 训练、target prior 更新、class-wise 评测与鲁棒评测流程均可在 AI Studio 上运行。

由于 CIFAR-100 类别数为 100，本实验只使用 5000 张训练图和 2 个 epoch，worst-class accuracy 为 0 属于预期现象。下一步应扩大训练规模到 20000 张和更多 epoch，用于观察有效性能趋势。
