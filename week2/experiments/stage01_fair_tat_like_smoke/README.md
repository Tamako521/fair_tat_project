# stage01_fair_tat_like_smoke

## 实验目的

本实验用于验证 FAIR-TAT-like CIFAR-10 训练入口是否跑通，包括：

- clean warmup
- class false positive prior 计算
- 基于 prior 的 target class 采样
- targeted PGD adversarial training
- class-wise robust accuracy 与 worst-class robust accuracy 评测

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py --epochs 2 --train-size 256 --test-size 128 --attack-eval-size 64 --batch-size 64 --pgd-steps 2 --output-dir week2\experiments\stage01_fair_tat_like_smoke
```

## 关键结果

```text
最终 clean accuracy: 0.1172
最终 clean worst-class accuracy: 0.0000
FGSM robust accuracy: 0.1094
FGSM worst-class robust accuracy: 0.0000
targeted PGD original accuracy: 0.1094
targeted PGD worst-class robust accuracy: 0.0000
targeted PGD target success rate: 0.5469
```

## 结果说明

该实验只使用很小的训练子集，主要价值是验证代码链路，不用于证明方法效果。由于样本少、训练轮次少，类别级 worst-class 指标为 0 是正常现象。更有意义的结果需要在更大的训练子集和更多 epoch 上比较。

## 已保存文件

- `train_log.txt`：训练与评测日志。
- `metrics.json`：完整指标、class-wise 指标和 target prior 历史。
- `train_fair_tat_like_cifar10.py`：本次实验使用的训练入口快照。
- `model.pth`：模型权重，默认不进入版本控制。
