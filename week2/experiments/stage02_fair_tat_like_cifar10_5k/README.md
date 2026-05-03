# stage02_fair_tat_like_cifar10_5k

## 实验目的

本实验是第二周 FAIR-TAT-like 复现实验的第一个可用于周报的小规模结果。相比 stage01 smoke test，本实验扩大到 CIFAR-10 训练子集 5000 张、测试子集 1000 张，用于观察 class false positive prior 采样的 targeted adversarial training 是否能正常训练并输出鲁棒公平性指标。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 3 --train-size 5000 --test-size 1000 --attack-eval-size 512 --batch-size 64 --pgd-steps 3 --prior-smoothing 5.0 --output-dir week2\experiments\stage02_fair_tat_like_cifar10_5k
```

## 关键结果

```text
最终 clean accuracy: 0.4660
最终 clean worst-class accuracy: 0.2584
FGSM robust accuracy: 0.1113
FGSM worst-class robust accuracy: 0.0000
targeted PGD original accuracy: 0.3613
targeted PGD worst-class robust accuracy: 0.1765
targeted PGD target success rate: 0.3047
```

## 初步观察

- 训练入口可以稳定完成 clean warmup、FAIR-TAT-like targeted adversarial training 和鲁棒评测。
- `target_prior` 会随 epoch 根据 confusion matrix 改变，说明 target class 不再是固定 `(label + 1) % 10`。
- targeted PGD original accuracy 达到 `0.3613`，已接近第一周固定目标 targeted PGD AT 小实验的 `0.3691`。
- 当前 FGSM worst-class robust accuracy 仍为 0，说明只做 targeted PGD 训练还不足以全面提升所有攻击下的最差类鲁棒性。

## 已保存文件

- `train_log.txt`：训练日志。
- `metrics.json`：完整评测结果。
- `train_fair_tat_like_cifar10.py`：本次实验训练入口快照。
- `model.pth`：模型权重，默认不进入版本控制。
