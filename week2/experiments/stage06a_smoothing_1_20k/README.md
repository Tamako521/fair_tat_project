# stage06a_smoothing_1_20k

## 实验目的

本实验测试 `prior_smoothing=1.0` 时 target prior 更尖锐的情况，观察更强调 false positive 类别是否能提升 worst-class robust accuracy。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 5 --train-size 20000 --test-size 2000 --attack-eval-size 1024 --batch-size 64 --pgd-steps 5 --adv-weight 0.5 --prior-smoothing 1.0 --output-dir week2\experiments\stage06a_smoothing_1_20k
```

## 关键结果

```text
clean accuracy: 0.5890
clean worst-class accuracy: 0.3280
FGSM robust accuracy: 0.1738
FGSM worst-class robust accuracy: 0.0556
targeted PGD original accuracy: 0.4072
targeted PGD worst-class robust accuracy: 0.2000
target success rate: 0.3633
```

## 结论

`prior_smoothing=1.0` 与 20k 默认设置接近，FGSM worst-class robust accuracy 略有提高，但 targeted PGD worst-class robust accuracy 没有明显提升。
