# stage05b_adv_weight_07_20k

## 实验目的

本实验用于测试 `adv_weight=0.7` 时 FAIR-TAT-like 训练的表现。该设置提高 adversarial loss 权重，更偏向 targeted PGD 鲁棒性。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 5 --train-size 20000 --test-size 2000 --attack-eval-size 1024 --batch-size 64 --pgd-steps 5 --adv-weight 0.7 --prior-smoothing 5.0 --output-dir week2\experiments\stage05b_adv_weight_07_20k
```

## 关键结果

```text
最终 clean accuracy: 0.5690
最终 clean worst-class accuracy: 0.3016
FGSM robust accuracy: 0.1943
FGSM worst-class robust accuracy: 0.0485
targeted PGD original accuracy: 0.4209
targeted PGD worst-class robust accuracy: 0.2222
targeted PGD target success rate: 0.3340
```

## 结论

`adv_weight=0.7` 的 targeted PGD original accuracy 和 targeted PGD worst-class robust accuracy 均优于 20k 默认 `adv_weight=0.5`。但它仍未超过 full 基线的 targeted PGD worst-class robust accuracy `0.2621`，因此暂不单独触发 full 复训。
