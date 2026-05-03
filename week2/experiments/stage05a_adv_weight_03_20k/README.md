# stage05a_adv_weight_03_20k

## 实验目的

本实验用于测试 `adv_weight=0.3` 时 FAIR-TAT-like 训练的表现。该设置降低 adversarial loss 权重，更偏向保持 clean accuracy。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 5 --train-size 20000 --test-size 2000 --attack-eval-size 1024 --batch-size 64 --pgd-steps 5 --adv-weight 0.3 --prior-smoothing 5.0 --output-dir week2\experiments\stage05a_adv_weight_03_20k
```

## 关键结果

```text
最终 clean accuracy: 0.5995
最终 clean worst-class accuracy: 0.3333
FGSM robust accuracy: 0.1455
FGSM worst-class robust accuracy: 0.0485
targeted PGD original accuracy: 0.3740
targeted PGD worst-class robust accuracy: 0.1889
targeted PGD target success rate: 0.4199
```

## 结论

`adv_weight=0.3` 提高了 clean accuracy，但 targeted PGD 鲁棒性弱于默认 `adv_weight=0.5` 和 `adv_weight=0.7`。该参数暂不适合作为 full 复训候选。
