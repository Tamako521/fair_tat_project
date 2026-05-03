# stage08a_adv07_smoothing10_full

## 实验目的

本实验使用完整 CIFAR-10 训练集测试组合参数：`adv_weight=0.7`、`prior_smoothing=10.0`、`PGD-5`。该组合来自 stage05 和 stage06 的观察：更高 adversarial loss 权重和更平滑 target prior 可能提高 targeted PGD 鲁棒性。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 6 --train-size 0 --test-size 5000 --attack-eval-size 2048 --batch-size 64 --pgd-steps 5 --adv-weight 0.7 --prior-smoothing 10.0 --output-dir week2\experiments\stage08a_adv07_smoothing10_full
```

## 关键结果

```text
clean accuracy: 0.6558
clean worst-class accuracy: 0.4686
FGSM robust accuracy: 0.2090
FGSM worst-class robust accuracy: 0.0806
targeted PGD original accuracy: 0.4761
targeted PGD worst-class robust accuracy: 0.2961
target success rate: 0.3184
```

## 结论

该组合是当前最佳候选。相比 stage04 full 基线，targeted PGD worst-class robust accuracy 从 `0.2621` 提升到 `0.2961`，FGSM worst-class robust accuracy 从 `0.0650` 提升到 `0.0806`。虽然 clean accuracy 略降，但仍保持在 `0.6558`。
