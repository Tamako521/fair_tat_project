# stage03_fair_tat_like_cifar10_20k

## 实验目的

本实验将 FAIR-TAT-like 训练规模扩大到 CIFAR-10 训练子集 20000 张，并将训练轮数提高到 5 个 epoch。相比 stage02 的 5k 小规模结果，本实验更适合作为第二周的主要实验结果。

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 5 --train-size 20000 --test-size 2000 --attack-eval-size 1024 --batch-size 64 --pgd-steps 5 --prior-smoothing 5.0 --output-dir week2\experiments\stage03_fair_tat_like_cifar10_20k
```

## 关键结果

```text
最终 clean accuracy: 0.5830
最终 clean worst-class accuracy: 0.3175
FGSM robust accuracy: 0.1748
FGSM worst-class robust accuracy: 0.0388
targeted PGD original accuracy: 0.4082
targeted PGD worst-class robust accuracy: 0.2000
targeted PGD target success rate: 0.3594
```

## 结果说明

训练规模提高后，模型的 clean accuracy 和 targeted PGD robust accuracy 都明显提升。尤其是 targeted PGD worst-class robust accuracy 达到 `0.2000`，说明动态 target prior 训练在更大数据规模下可以产生可观察的类别级鲁棒性结果。

FGSM worst-class robust accuracy 仍然偏低，但已经从 stage02 的 `0.0000` 提升到 `0.0388`。这说明扩大数据和训练轮数对最差类鲁棒性有帮助，但 targeted PGD 训练还不能完全解决不同攻击类型下的类别级不平衡。

## 已保存文件

- `train_log.txt`：训练日志。
- `metrics.json`：完整评测结果。
- `train_fair_tat_like_cifar10.py`：本次实验训练入口快照。
- `model.pth`：模型权重，默认不进入版本控制。
