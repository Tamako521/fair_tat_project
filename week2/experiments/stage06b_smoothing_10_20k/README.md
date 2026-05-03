# stage06b_smoothing_10_20k

## 实验目的

本实验测试 `prior_smoothing=10.0` 时 target prior 更平滑的情况，观察降低 target prior 尖锐程度是否能改善训练稳定性。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 5 --train-size 20000 --test-size 2000 --attack-eval-size 1024 --batch-size 64 --pgd-steps 5 --adv-weight 0.5 --prior-smoothing 10.0 --output-dir week2\experiments\stage06b_smoothing_10_20k
```

## 关键结果

```text
clean accuracy: 0.5880
clean worst-class accuracy: 0.3333
FGSM robust accuracy: 0.1836
FGSM worst-class robust accuracy: 0.0485
targeted PGD original accuracy: 0.4170
targeted PGD worst-class robust accuracy: 0.2182
target success rate: 0.3584
```

## 结论

`prior_smoothing=10.0` 的 targeted PGD worst-class robust accuracy 高于默认 20k 和 `smoothing=1.0`，说明更平滑的 target prior 对 targeted PGD 最差类鲁棒性略有帮助。
