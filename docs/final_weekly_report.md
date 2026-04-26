# 周报：FAIR-TAT 复现前置实验与对抗训练初步验证

姓名：郭家棋

## 一、本周工作概述

本周围绕 FAIR-TAT 复现前置工作，完成了基础环境搭建、CIFAR-10 数据加载、基础训练流程验证、对抗攻击模块接入、对抗训练初步实验以及类别级鲁棒性分析。整体工作从最小 smoke test 开始，逐步扩展到 10k 训练规模下的 clean baseline、FGSM adversarial training 和 targeted PGD adversarial training 对比实验。

本周重点不是完成 FAIR-TAT 的完整复现，而是搭建后续复现所需的基础实验链路，包括：

1. 基础分类训练流程。
2. FGSM 与 targeted PGD 对抗样本生成。
3. clean accuracy、adversarial accuracy、worst-class accuracy 等指标统计。
4. 不同训练方式下的鲁棒性对比。
5. 类别级表现差异分析。

## 二、阶段性工作

### 1. 环境搭建与最小训练流程验证

首先完成了项目基础环境搭建，并验证 PyTorch、torchvision、CIFAR-10 数据加载流程可以正常运行。随后使用一个简单的 `SimpleCNN` 完成单 batch smoke test，验证了模型前向传播、交叉熵损失计算、反向传播、优化器更新和模型保存流程。

初始输出示例：

```text
loss: 2.2881388664245605
训练成功！
```

之后进一步在该流程上接入 FGSM 与 targeted PGD 攻击，验证对抗样本生成流程可以与训练代码衔接。

### 2. 对抗攻击流程验证

在基础训练流程跑通后，实现并验证了两类攻击：

- FGSM untargeted attack。
- targeted PGD attack。

其中 targeted PGD 暂时使用简单规则构造目标类别：

```text
target_label = (label + 1) % 10
```

该规则不是 FAIR-TAT 的最终 target class 选择策略，只用于验证 targeted adversarial attack 流程。

一次 smoke test 输出如下：

```text
device: cpu
clean loss: 2.3192856311798096
fgsm adversarial loss: 2.343320846557617
clean target loss: 2.8706908226013184
targeted pgd loss to target: 2.568723678588867
targeted pgd loss to original: 2.099764108657837
训练成功！
FGSM 对抗攻击流程验证成功！
targeted PGD 对抗攻击流程验证成功！
```

其中 target loss 从 `2.8707` 降到 `2.5687`，说明 targeted PGD 的优化方向能够推动样本向指定目标类别靠近。

### 3. 小规模 baseline 与对抗训练实验

随后在 CIFAR-10 子集上完成了初步 baseline 和对抗训练实验。实验包括：

- clean baseline。
- FGSM adversarial training。
- targeted PGD adversarial training。

在 5k 训练样本、1k 测试样本的初步实验中，观察到 FGSM adversarial training 可以提升 FGSM adversarial accuracy，但会带来 clean accuracy 和 worst-class accuracy 的下降。这一现象与 FAIR-TAT 关注的对抗训练公平性问题有一定关联。

### 4. 类别级表现分析

为了进一步观察类别间表现差异，对 clean baseline 与 FGSM adversarial training 的类别级 clean accuracy 进行了分析。结果显示，简单 FGSM 对抗训练并不会均匀改善所有类别。

部分观察如下：

- 类别 6、7、9 在 FGSM adversarial training 后准确率上升。
- 类别 2、3、4 准确率明显下降。
- 类别 2 从 `0.2921` 下降到 `0.0112`，成为新的最差类别。

这说明对抗训练的收益可能在类别间分布不均，后续需要进一步研究 FAIR-TAT 中 targeted adversarial training 是否能够改善最差类别表现。

### 5. 对抗类别级鲁棒性分析

进一步分析了各类别在 FGSM 与 targeted PGD 攻击下的鲁棒性表现。结果显示：

- FGSM adversarial training 提升了多数类别的 FGSM adversarial accuracy。
- 类别 2 和类别 4 在 FGSM 攻击下仍然为 `0.0000`，说明它们仍是脆弱类别。
- targeted PGD 场景下，多数类别的目标攻击成功率下降，但部分类别的攻击成功率反而上升。

这一结果进一步说明，简单对抗训练对不同类别的鲁棒性提升并不均衡。

## 三、10k 训练规模主实验

为了获得更稳定的结果，本周进一步扩大训练规模，使用 10000 个 CIFAR-10 训练样本、2000 个测试样本和 5 个 epoch，对三种训练方式进行了同规模对比。

统一实验设置：

- 数据集：CIFAR-10
- 训练样本数：10000
- 测试样本数：2000
- 对抗评测样本数：1000
- 模型：SmallCNN
- 训练轮数：5
- batch size：64
- 优化器：Adam
- 学习率：0.001
- FGSM epsilon：8 / 255
- targeted PGD epsilon：8 / 255
- targeted PGD alpha：2 / 255
- targeted PGD steps：5
- 设备：CPU

主实验结果如下：

| 实验 | clean accuracy | worst-class accuracy | FGSM adversarial accuracy | targeted PGD original accuracy | targeted PGD target success rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean baseline | 0.5750 | 0.3333 | 0.0684 | 0.2051 | 0.5918 |
| FGSM adversarial training | 0.5120 | 0.2646 | 0.2246 | 0.3945 | 0.2119 |
| targeted PGD adversarial training | 0.4870 | 0.2629 | 0.1201 | 0.4756 | 0.2998 |

## 四、结果分析

从 10k 训练规模实验可以看到：

1. clean baseline 的 clean accuracy 最高，达到 `0.5750`，但 FGSM adversarial accuracy 只有 `0.0684`，说明普通训练模型对 FGSM 扰动较脆弱。
2. FGSM adversarial training 将 FGSM adversarial accuracy 从 `0.0684` 提升到 `0.2246`，但 clean accuracy 从 `0.5750` 下降到 `0.5120`。
3. targeted PGD adversarial training 的 targeted PGD original accuracy 达到 `0.4756`，是三组中最高，说明它对当前 targeted PGD 攻击更有针对性。
4. 两类对抗训练的 worst-class accuracy 都低于 clean baseline，说明对抗训练后的类别级最差表现仍需进一步优化。
5. 不同对抗训练方式提升的鲁棒性类型不同，FGSM adversarial training 更明显提升 FGSM 鲁棒性，而 targeted PGD adversarial training 更明显提升 targeted PGD 场景下的原类别保持率。

这些现象与 FAIR-TAT 的研究动机一致：对抗训练不能只关注整体鲁棒性，还需要关注不同类别之间的鲁棒性差异和最差类别表现。

## 五、当前问题

1. 当前实验仍然基于 SmallCNN 和 CIFAR-10 子集，不能代表最终复现结果。
2. targeted PGD 目标类别目前使用 `(label + 1) % 10` 的简单规则，后续需要对齐 FAIR-TAT 论文中的 target class 选择策略。
3. 目前 worst-class accuracy 只是类别级最差准确率统计，后续需要继续扩展 class-wise robust accuracy 等更贴近 FAIR-TAT 的指标。
4. CIFAR-10 缺少天然敏感属性，更适合用于攻击流程和鲁棒性流程验证；医疗图像数据集上的公平性分析仍需进一步开展。
5. ISIC 2019 等医疗图像数据集的数据划分、标签格式和可用属性仍需继续整理。

## 六、下周计划

1. 阅读 FAIR-TAT 方法部分，重点梳理论文中的 target class 选择策略和训练损失设计。
2. 将当前简单的 `(label + 1) % 10` 目标类别规则替换为更接近 FAIR-TAT 的目标类别选择方式。
3. 继续完善 class-wise robust accuracy、worst-class robust accuracy 等评测指标。
4. 在统一设置下进一步比较 clean baseline、FGSM adversarial training、targeted PGD adversarial training 和 FAIR-TAT 近似实现。
5. 整理代码结构，将攻击函数、训练入口和评测函数拆分为更清晰的模块。
6. 调研 ISIC 2019 等医疗图像数据集的标签、属性和公平性评测设置，为后续医疗图像实验做准备。

## 七、实验材料位置

本周实验材料已整理到项目目录中：

- smoke test：`experiments/stage01_smoke_test`
- 5k clean baseline：`experiments/stage02_cifar10_baseline_cpu_5k`
- 5k FGSM adversarial training：`experiments/stage03_fgsm_adversarial_training_cpu_5k`
- 5k targeted PGD adversarial training：`experiments/stage04_targeted_pgd_adversarial_training_cpu_5k`
- 10k clean baseline：`experiments/stage05_cifar10_baseline_cpu_10k`
- 10k FGSM adversarial training：`experiments/stage06_fgsm_adversarial_training_cpu_10k`
- 10k targeted PGD adversarial training：`experiments/stage07_targeted_pgd_adversarial_training_cpu_10k`
- 10k 对比总结：`docs/experiment_comparison_10k.md`
- 类别级分析：`docs/classwise_analysis.md`
- 对抗类别级分析：`docs/adversarial_classwise_analysis.md`

