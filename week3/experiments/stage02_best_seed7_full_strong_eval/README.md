# stage02_best_seed7_full_strong_eval

## 目的

使用第二周筛出的最佳 FAIR-TAT-like 参数在 full CIFAR-10 上复现实验，并加入更强评测攻击，检查结果稳定性。

## 配置

- 模型：SmallCNN
- 训练数据：full CIFAR-10 train
- 测试数据：CIFAR-10 test 子集 5000
- 训练轮数：6
- 随机种子：7
- 训练攻击：targeted PGD-5
- 评测攻击：FGSM、multi-step FGSM-5、untargeted PGD-10、targeted PGD-5
- `adv_weight=0.7`
- `prior_smoothing=10.0`

## 结果摘要

- clean accuracy：0.6332
- clean worst-class accuracy：0.3280
- FGSM robust accuracy：0.2271
- FGSM worst-class robust accuracy：0.0311
- multi-step FGSM robust accuracy：0.1816
- multi-step FGSM worst-class robust accuracy：0.0267
- untargeted PGD robust accuracy：0.1694
- untargeted PGD worst-class robust accuracy：0.0222
- targeted PGD original accuracy：0.4717
- targeted PGD worst-class robust accuracy：0.2533
- targeted PGD target success rate：0.3042

## 初步观察

该配置在 targeted PGD 评测下仍能保持一定 class-wise robustness，但在 untargeted PGD-10 下 worst-class robust accuracy 明显偏低，说明后续需要加入更强的评测攻击和 per-class 鲁棒性调节。
