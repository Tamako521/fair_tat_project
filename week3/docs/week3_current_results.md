# 第三周当前实验结果

## 代码进展

本周已完成以下工程扩展：

- 在 `src.models` 中加入 `build_model()`，支持 `small_cnn` 和 `resnet18`。
- 在 `src.attacks` 中加入 multi-step FGSM 和 untargeted PGD。
- 在 `src.evaluation` 中加入 class-wise multi-step FGSM / untargeted PGD robust accuracy 和 worst-class 指标。
- 在训练脚本中加入 `--model`、`--eval-pgd-steps`、`--eval-ifgsm-steps` 参数。
- 训练脚本现在每轮保存 `model_latest.pth` 和 `history_latest.json`，避免断电后完全丢失阶段结果。

## 结果总表

| 实验 | 模型 | 训练规模 | seed | clean acc | clean worst | FGSM robust | IFGSM robust | untargeted PGD robust | targeted PGD original | targeted PGD worst |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage01 smoke | ResNet18 | 128 | 42 | 0.2188 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1250 | 0.0000 |
| stage02 best repeat | SmallCNN | full | 7 | 0.6332 | 0.3280 | 0.2271 | 0.1816 | 0.1694 | 0.4717 | 0.2533 |
| stage03 best repeat | SmallCNN | full | 123 | 0.6480 | 0.3464 | 0.2461 | 0.1997 | 0.1826 | 0.4956 | 0.2374 |
| stage04 ResNet18 probe | ResNet18 | 5k | 42 | 0.4470 | 0.0899 | 0.1582 | 0.1895 | 0.1641 | 0.3828 | 0.0784 |

## 初步结论

1. `adv_weight=0.7, prior_smoothing=10.0, PGD-5` 在两个 full CIFAR-10 seed 上可以稳定跑通，clean accuracy 约为 0.63-0.65。
2. targeted PGD original accuracy 在两个 seed 中分别为 0.4717 和 0.4956，说明第二周筛出的配置仍有一定稳定性。
3. untargeted PGD-10 明显更强，平均 robust accuracy 仅约 0.17-0.18，worst-class robust accuracy 低于 0.03，是后续优化重点。
4. ResNet18 入口已经验证可用，但 5k 数据和 2 epoch 训练不足，暂时不能据此判断更强模型是否优于 SmallCNN。
5. 下一步建议优先做 `stage05_resnet18_20k_or_more`，同时尝试 per-class robust deficit 调节 target prior。
