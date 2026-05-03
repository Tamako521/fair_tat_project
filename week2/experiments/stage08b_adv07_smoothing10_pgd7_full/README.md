# stage08b_adv07_smoothing10_pgd7_full

## 实验目的

本实验在 stage08a 的组合参数基础上进一步提高训练攻击强度，将 PGD 步数从 5 提高到 7。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 6 --train-size 0 --test-size 5000 --attack-eval-size 2048 --batch-size 64 --pgd-steps 7 --adv-weight 0.7 --prior-smoothing 10.0 --output-dir week2\experiments\stage08b_adv07_smoothing10_pgd7_full
```

## 关键结果

```text
clean accuracy: 0.6494
clean worst-class accuracy: 0.4529
FGSM robust accuracy: 0.2153
FGSM worst-class robust accuracy: 0.0885
targeted PGD original accuracy: 0.4844
targeted PGD worst-class robust accuracy: 0.2767
target success rate: 0.3159
```

## 结论

PGD-7 组合获得了当前最高的 targeted PGD original accuracy 和 FGSM worst-class robust accuracy，但 targeted PGD worst-class robust accuracy 低于 stage08a。因此如果优先目标是 FAIR-TAT 的 worst-class robust accuracy，stage08a 更适合作为当前最佳模型。
