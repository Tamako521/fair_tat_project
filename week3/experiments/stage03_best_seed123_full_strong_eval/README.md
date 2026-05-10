# stage03_best_seed123_full_strong_eval

## 目的

在 full CIFAR-10 上使用第二个随机种子复现当前最佳 FAIR-TAT-like 参数，检查结果是否稳定。

## 配置

- 模型：SmallCNN
- 训练数据：full CIFAR-10 train
- 测试数据：CIFAR-10 test 子集 5000
- 训练轮数：6
- 随机种子：123
- 训练攻击：targeted PGD-5
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-10、targeted PGD-5
- `adv_weight=0.7`
- `prior_smoothing=10.0`

## 结果摘要

- clean accuracy：0.6480
- clean worst-class accuracy：0.3464
- FGSM robust accuracy：0.2461
- FGSM worst-class robust accuracy：0.0556
- multi-step FGSM robust accuracy：0.1997
- multi-step FGSM worst-class robust accuracy：0.0297
- untargeted PGD robust accuracy：0.1826
- untargeted PGD worst-class robust accuracy：0.0297
- targeted PGD original accuracy：0.4956
- targeted PGD worst-class robust accuracy：0.2374
- targeted PGD target success rate：0.2983

## 初步观察

相比 seed 7，该随机种子下 clean accuracy 和平均鲁棒性略高，但 targeted PGD worst-class robust accuracy 略低。两个 seed 都显示：平均鲁棒性能维持，但 worst-class 指标仍是主要短板。
