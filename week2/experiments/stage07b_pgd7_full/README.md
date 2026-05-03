# stage07b_pgd7_full

## 实验目的

本实验直接使用完整 CIFAR-10 训练集测试 PGD-7 训练强度，观察更强 targeted PGD 训练是否超过 stage04 full 基线。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 6 --train-size 0 --test-size 5000 --attack-eval-size 2048 --batch-size 64 --pgd-steps 7 --adv-weight 0.5 --prior-smoothing 5.0 --output-dir week2\experiments\stage07b_pgd7_full
```

## 关键结果

```text
clean accuracy: 0.6656
clean worst-class accuracy: 0.4686
FGSM robust accuracy: 0.1919
FGSM worst-class robust accuracy: 0.0650
targeted PGD original accuracy: 0.4624
targeted PGD worst-class robust accuracy: 0.2573
target success rate: 0.3555
```

## 结论

PGD-7 提高了 targeted PGD original accuracy，但 targeted PGD worst-class robust accuracy 略低于 stage04 full 基线，因此不是当前最优参数。
