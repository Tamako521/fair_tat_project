# AI Studio stage01 任务说明

## 一、需要同步到 AI Studio 的内容

建议直接同步以下内容，避免本地和云端代码版本不一致：

- `src/`
- `scripts/train_fair_tat_like_cifar10.py`
- `week5/docs/week5_task_plan.md`
- `week5/docs/aistudio_stage01.md`

数据部分：

- 如果 AI Studio 能联网，不需要手动上传 CIFAR-100，脚本会自动下载。
- 如果 AI Studio 下载失败，上传本地 `data/cifar-100-python.tar.gz` 和解压后的 `data/cifar-100-python/`。

## 二、stage01 实验目的

在 AI Studio GPU 上验证：

- CIFAR-100 能正常训练；
- ResNet18 的 100 类输出正常；
- FAIR-TAT-like 的 blended target prior、targeted PGD、robust deficit loss weight 能在 CIFAR-100 上跑通；
- 指标文件、日志、模型权重能正常保存。

stage01 仍然是验证实验，不追求最终精度。CIFAR-100 难度明显高于 CIFAR-10，前几轮准确率偏低是正常现象。

## 三、stage01 推荐命令

```bash
python scripts/train_fair_tat_like_cifar10.py \
  --dataset cifar100 \
  --output-dir week5/experiments/stage01_cifar100_resnet18_smoke \
  --epochs 2 \
  --batch-size 256 \
  --num-workers 8 \
  --pin-memory \
  --amp \
  --train-size 5000 \
  --test-size 1000 \
  --attack-eval-size 512 \
  --learning-rate 0.001 \
  --model resnet18 \
  --seed 42 \
  --pgd-steps 2 \
  --eval-pgd-steps 3 \
  --eval-ifgsm-steps 2 \
  --adv-weight 0.8 \
  --warmup-epochs 1 \
  --prior-smoothing 10.0 \
  --prior-mode blended \
  --deficit-source targeted_pgd \
  --deficit-smoothing 1.0 \
  --deficit-weight 0.5 \
  --prior-eval-size 256 \
  --loss-weight-mode robust_deficit \
  --loss-weight-strength 1.0
```

## 四、stage01 完成后检查

查看以下文件：

- `week5/experiments/stage01_cifar100_resnet18_smoke/train_log.txt`
- `week5/experiments/stage01_cifar100_resnet18_smoke/metrics.json`
- `week5/experiments/stage01_cifar100_resnet18_smoke/model.pth`

重点记录：

- final clean accuracy
- final clean worst-class accuracy
- untargeted PGD robust accuracy
- untargeted PGD worst-class robust accuracy
- targeted PGD original accuracy
- targeted PGD worst-class robust accuracy
- targeted PGD target success rate

## 五、stage01 后续判断

如果 stage01 跑通：

1. 进入 `stage02_cifar100_resnet18_20k`，扩大到 `train-size 20000`、`epochs 4`。
2. 若 stage02 指标正常，再进入 `stage03_cifar100_resnet18_full`，使用完整 CIFAR-100。
3. 如果 stage01 速度很快，可以直接把 `batch-size` 提到 512 或 1024。
