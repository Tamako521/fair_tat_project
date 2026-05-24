# week5_local_cifar100_smoke

## 实验目的

本地最小测试 CIFAR-100 数据集接入是否成功，确认以下流程可用：

- `--dataset cifar100` 参数解析
- CIFAR-100 下载与读取
- 100 类分类头初始化
- targeted PGD 训练入口
- clean / FGSM / IFGSM / untargeted PGD / targeted PGD 评测

## 运行命令

```powershell
.\.venv\Scripts\python.exe scripts\train_fair_tat_like_cifar10.py `
  --dataset cifar100 `
  --output-dir week5\experiments\week5_local_cifar100_smoke `
  --epochs 1 `
  --batch-size 16 `
  --train-size 128 `
  --test-size 64 `
  --attack-eval-size 32 `
  --learning-rate 0.001 `
  --model small_cnn `
  --seed 42 `
  --pgd-steps 1 `
  --eval-pgd-steps 1 `
  --eval-ifgsm-steps 1 `
  --adv-weight 0.7 `
  --warmup-epochs 0 `
  --prior-smoothing 10.0 `
  --prior-mode blended `
  --deficit-source targeted_pgd `
  --deficit-smoothing 1.0 `
  --deficit-weight 0.5 `
  --prior-eval-size 32 `
  --loss-weight-mode robust_deficit `
  --loss-weight-strength 1.0
```

## 测试结果

本测试已成功完成，指标保存在 `metrics.json`。

由于该实验只使用 128 张训练图、64 张测试图，并且 CIFAR-100 有 100 个类别，因此准确率不作为有效实验结论，仅作为数据入口和训练评测流程验证。
