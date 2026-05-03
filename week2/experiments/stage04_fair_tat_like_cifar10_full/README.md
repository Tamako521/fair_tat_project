# stage04_fair_tat_like_cifar10_full

## 实验目的

本实验使用完整 CIFAR-10 训练集运行 FAIR-TAT-like 训练，是第二周当前最主要的实验结果。相比 5k 和 20k 实验，本实验进一步扩大训练数据，并提高测试与对抗评测规模。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 6 --train-size 0 --test-size 5000 --attack-eval-size 2048 --batch-size 64 --pgd-steps 5 --prior-smoothing 5.0 --output-dir week2\experiments\stage04_fair_tat_like_cifar10_full
```

说明：`train-size=0` 表示使用完整 CIFAR-10 训练集。

## 关键结果

```text
最终 clean accuracy: 0.6670
最终 clean worst-class accuracy: 0.4843
FGSM robust accuracy: 0.1919
FGSM worst-class robust accuracy: 0.0650
targeted PGD original accuracy: 0.4561
targeted PGD worst-class robust accuracy: 0.2621
targeted PGD target success rate: 0.3481
```

## 结果说明

扩大到完整 CIFAR-10 训练集后，clean accuracy 达到 `0.6670`，clean worst-class accuracy 达到 `0.4843`。这说明模型的自然分类性能和类别级最差表现都明显优于前两个 FAIR-TAT-like 阶段实验。

targeted PGD original accuracy 达到 `0.4561`，targeted PGD worst-class robust accuracy 达到 `0.2621`。相比 20k 实验的 `0.4082` 和 `0.2000`，全量训练进一步提升了对 targeted PGD 的整体鲁棒性和最差类鲁棒性。

FGSM worst-class robust accuracy 仍然偏低，为 `0.0650`。这提示当前训练主要提升了 targeted PGD 对应威胁模型下的鲁棒性，但跨攻击类型的最差类鲁棒性仍需继续优化。

## 已保存文件

- `train_log.txt`：训练日志。
- `metrics.json`：完整评测结果。
- `train_fair_tat_like_cifar10.py`：本次实验训练入口快照。
- `model.pth`：模型权重，默认不进入版本控制。
