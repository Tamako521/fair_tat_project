# 周报：FAIR-TAT 复现前置实验

姓名：郭家棋

## 一、本周完成工作

本周主要完成了 FAIR-TAT 复现前的实验框架搭建和初步验证，重点包括：

1. 跑通 PyTorch、torchvision 和 CIFAR-10 数据加载流程。
2. 完成基础 `SimpleCNN` / `SmallCNN` 训练流程，验证前向传播、损失计算、反向传播和模型保存。
3. 实现并验证 FGSM untargeted attack 和 targeted PGD attack。
4. 完成 clean baseline、FGSM adversarial training 和 targeted PGD adversarial training 三类实验。
5. 增加 clean accuracy、FGSM adversarial accuracy、targeted PGD original accuracy、worst-class accuracy 等评测指标。
6. 初步分析不同训练方式下的类别级表现差异，为后续 FAIR-TAT 的鲁棒公平性分析做准备。

## 二、实验顺序

实验按以下顺序推进：

1. `stage01_smoke_test`：验证环境、数据加载、基础训练、FGSM 和 targeted PGD 攻击流程。
2. `stage02_cifar10_baseline_cpu_5k`：在 5k CIFAR-10 子集上训练 clean baseline。
3. `stage03_fgsm_adversarial_training_cpu_5k`：验证 FGSM 对抗训练入口。
4. `stage04_targeted_pgd_adversarial_training_cpu_5k`：验证 targeted PGD 对抗训练入口。
5. `stage05_cifar10_baseline_cpu_10k`：扩大到 10k 训练规模，得到主实验 baseline。
6. `stage06_fgsm_adversarial_training_cpu_10k`：同规模下比较 FGSM adversarial training。
7. `stage07_targeted_pgd_adversarial_training_cpu_10k`：同规模下比较 targeted PGD adversarial training。

## 三、主要实验结果

10k 训练规模实验设置：

- 数据集：CIFAR-10
- 训练样本数：10000
- 测试样本数：2000
- 对抗评测样本数：1000
- 模型：SmallCNN
- 训练轮数：5
- 设备：CPU

| 实验 | clean accuracy | worst-class accuracy | FGSM adversarial accuracy | targeted PGD original accuracy | targeted PGD target success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean baseline | 0.5750 | 0.3333 | 0.0684 | 0.2051 | 0.5918 |
| FGSM adversarial training | 0.5120 | 0.2646 | 0.2246 | 0.3945 | 0.2119 |
| targeted PGD adversarial training | 0.4870 | 0.2629 | 0.1201 | 0.4756 | 0.2998 |

## 四、初步分析

1. clean baseline 的 clean accuracy 最高，但对 FGSM 和 targeted PGD 攻击较脆弱。
2. FGSM adversarial training 将 FGSM adversarial accuracy 从 `0.0684` 提升到 `0.2246`，但 clean accuracy 下降到 `0.5120`。
3. targeted PGD adversarial training 的 targeted PGD original accuracy 最高，达到 `0.4756`，说明它对 targeted PGD 攻击更有针对性。
4. 两类对抗训练的 worst-class accuracy 都低于 clean baseline，说明简单对抗训练可能带来类别级最差表现下降。
5. 类别级分析显示，不同类别的收益不均衡，部分类别鲁棒性提升明显，部分类别仍较脆弱。这与 FAIR-TAT 关注的类别级鲁棒公平性问题一致。

## 五、当前问题

1. 当前模型仍是 SmallCNN，实验还属于前置验证，不能代表最终复现效果。
2. targeted PGD 的目标类别暂时使用 `(label + 1) % 10`，后续需要对齐 FAIR-TAT 论文中的 target class 选择策略。
3. 当前 worst-class accuracy 只是初步类别级指标，后续需要扩展到 class-wise robust accuracy。
4. CIFAR-10 缺少敏感属性，后续医疗图像数据集实验仍需单独设计公平性评测。

## 六、下周计划

1. 继续阅读 FAIR-TAT 方法部分，重点整理 target class 选择策略和训练损失。
2. 将当前 targeted PGD 训练入口改造成更接近 FAIR-TAT 的版本。
3. 完善 class-wise robust accuracy 和 worst-class robust accuracy 评测。
4. 整理代码结构，将攻击、训练和评测模块拆分得更清晰。
5. 调研 ISIC 2019 等医疗图像数据集的标签、属性和公平性评测设置。

## 七、材料位置

实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：

- 实验顺序说明：`docs/experiment_order.md`
- 10k 对比总结：`docs/experiment_comparison_10k.md`
- 类别级分析：`docs/classwise_analysis.md`
- 对抗类别级分析：`docs/adversarial_classwise_analysis.md`
- 实验结果目录：`experiments/stage01_smoke_test` 至 `experiments/stage07_targeted_pgd_adversarial_training_cpu_10k`
