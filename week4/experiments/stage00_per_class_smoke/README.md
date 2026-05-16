# stage00_per_class_smoke

## 目的

验证第四周新增的 per-class robust deficit target prior 和 class-weighted adversarial loss 可以正常训练、评测和落盘。

## 配置

- 模型：SmallCNN
- 训练数据：CIFAR-10 子集 256
- 测试数据：CIFAR-10 子集 128
- 训练轮数：2
- `prior_mode=blended`
- `deficit_source=targeted_pgd`
- `loss_weight_mode=robust_deficit`

## 结论

该实验只用于流程验证。日志中已经能看到 `deficit_prior`、`target_prior` 和 `class_loss_weights`，说明第四周新增调节逻辑已经接入训练管线。
