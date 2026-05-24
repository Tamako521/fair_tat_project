# 周报：CIFAR-100 数据集扩展与训练稳定性验证

姓名：郭家棋

## 一、本周完成工作

本周在前期 CIFAR-10 实验基础上，开始扩展训练数据集，重点接入 CIFAR-100 并验证 FAIR-TAT-like 方法在更难数据集上的可行性。

1. 将数据加载和训练入口从 CIFAR-10 扩展到 CIFAR-100，新增 `--dataset cifar100` 参数。
2. 完成本地 CIFAR-100 最小测试，确认数据下载、100 类分类头、targeted PGD 训练和鲁棒评测流程可用。
3. 在百度 AI Studio Tesla V100 上进行 CIFAR-100 训练实验。
4. 定位 AI Studio 环境中 AMP / cuDNN 会导致段错误的问题，最终采用 `--disable-cudnn`、不使用 AMP 的稳定配置。
5. 完成 CIFAR-100 的 5k、20k、full probe 和类别权重增强实验。

## 二、主要实验结果

主要设置：

- 数据集：CIFAR-100
- 模型：ResNet18
- 训练攻击：targeted PGD
- 评测攻击：FGSM、multi-step FGSM、untargeted PGD、targeted PGD
- 平台：百度 AI Studio，Tesla V100 GPU

| 实验 | 设置 | clean acc | clean worst | untargeted PGD | targeted PGD original | targeted PGD worst |
|---|---|---:|---:|---:|---:|---:|
| stage01 | 5k / 2 epochs | 0.0980 | 0.0000 | 0.0547 | 0.1055 | 0.0000 |
| stage02 | 20k / 4 epochs | 0.3297 | 0.0000 | 0.0811 | 0.2773 | 0.0000 |
| stage03 | full / 3 epochs / fast probe | **0.3908** | **0.0179** | **0.0898** | **0.3516** | 0.0000 |
| stage04 | 20k / stronger class weight | 0.3163 | 0.0000 | 0.0840 | 0.2676 | 0.0000 |

## 三、结果分析

1. CIFAR-100 已成功接入，训练和评测流程均能跑通。
2. 随着训练数据从 5k 扩大到 20k 和 full，clean accuracy 明显提升，说明模型确实在 CIFAR-100 上有效学习。
3. CIFAR-100 比 CIFAR-10 更难，worst-class 指标仍然偏低，尤其是 targeted PGD worst-class 仍为 0。
4. 简单增强 class-weighted loss 没有明显改善 worst-class，说明后续需要更细的 target class 更新或分组鲁棒性调节策略。
5. AI Studio GPU 能支持更大规模实验，但当前环境下 AMP / cuDNN 不稳定，训练速度受到一定影响。

## 四、本周结论

本周完成了从 CIFAR-10 到 CIFAR-100 的数据集扩展，并完成多组验证实验。结果表明，当前 FAIR-TAT-like 方法可以迁移到 CIFAR-100，但在 100 类任务上最差类鲁棒性仍是主要瓶颈。

## 五、下周计划

1. 优化 CIFAR-100 的 target class 更新策略，重点改善 worst-class robust accuracy。
2. 尝试降低评测开销，提高实验迭代速度。
3. 对比不同 `adv_weight`、`loss_weight_strength` 和 `deficit_weight` 的影响。
4. 继续准备医疗图像数据集接入，为 ISIC 2019 公平性评测做铺垫。

## 六、材料位置
实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：
- 主要实验目录：`week5/experiments/stage01_cifar100_resnet18_safe` 至 `week5/experiments/stage04_cifar100_resnet18_20k_stronger_class_weight`
