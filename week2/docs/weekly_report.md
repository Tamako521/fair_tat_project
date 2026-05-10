# 周报：FAIR-TAT-like 复现与参数优化实验

姓名：郭家棋

## 一、本周完成工作

本周主要围绕 FAIR-TAT 方法复现推进，重点从第一周的固定目标类别 targeted PGD，对齐到更接近 FAIR-TAT 思想的动态 target class 选择策略，并完成了多组 CIFAR-10 实验。

1. 继续阅读 FAIR-TAT 方法部分，整理了 targeted adversarial training、target class prior 和训练损失设计。
2. 将原本固定 `(label + 1) % 10` 的目标类别选择，改为基于 confusion matrix 的 class false positive prior 采样。
3. 新增 FAIR-TAT-like CIFAR-10 训练入口，支持 clean warmup、target prior 更新、targeted PGD 训练和鲁棒评测。
4. 整理代码结构，将模型、攻击、数据加载、target selection、训练和评测拆分到 `src/` 模块中。
5. 完善评测指标，增加 FGSM / targeted PGD 的 class-wise robust accuracy 和 worst-class robust accuracy。
6. 完成从 5k、20k 到完整 CIFAR-10 训练集的逐步扩大实验。
7. 进一步完成 `adv_weight`、`prior_smoothing`、PGD steps 等参数优化实验，并得到当前最优组合。

## 二、实验顺序

本周实验按以下顺序推进：

1. `stage01_fair_tat_like_smoke`：验证 FAIR-TAT-like 新入口能否完整跑通。
2. `stage02_fair_tat_like_cifar10_5k`：在 5k CIFAR-10 子集上做小规模验证。
3. `stage03_fair_tat_like_cifar10_20k`：扩大到 20k 训练数据，观察指标是否稳定提升。
4. `stage04_fair_tat_like_cifar10_full`：使用完整 CIFAR-10 训练集得到 full 基线。
5. `stage05a / stage05b`：比较 `adv_weight=0.3` 和 `adv_weight=0.7`。
6. `stage06a / stage06b`：比较 `prior_smoothing=1.0` 和 `prior_smoothing=10.0`。
7. `stage07b_pgd7_full`：在完整训练集上测试 PGD-7。
8. `stage08a / stage08b`：组合较优参数后进行 full 复训，得到当前最佳结果。

## 三、主要实验结果

主要实验设置：

- 数据集：CIFAR-10
- 模型：SmallCNN
- 攻击：FGSM、targeted PGD
- `epsilon=8/255`
- `alpha=2/255`
- 设备：CPU

| 实验 | clean acc | clean worst-class acc | FGSM robust acc | FGSM worst-class robust acc | targeted PGD original acc | targeted PGD worst-class robust acc | target success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| week2 FAIR-TAT-like 5k | 0.4660 | 0.2584 | 0.1113 | 0.0000 | 0.3613 | 0.1765 | 0.3047 |
| week2 FAIR-TAT-like 20k | 0.5830 | 0.3175 | 0.1748 | 0.0388 | 0.4082 | 0.2000 | 0.3594 |
| week2 FAIR-TAT-like full | 0.6670 | 0.4843 | 0.1919 | 0.0650 | 0.4561 | 0.2621 | 0.3481 |
| stage08a adv0.7 smooth10 full | 0.6558 | 0.4686 | 0.2090 | 0.0806 | 0.4761 | **0.2961** | 0.3184 |
| stage08b adv0.7 smooth10 PGD-7 full | 0.6494 | 0.4529 | 0.2153 | **0.0885** | **0.4844** | 0.2767 | 0.3159 |

## 四、结果分析

1. FAIR-TAT-like 训练从 5k 扩大到 full 后，clean accuracy 从 `0.4660` 提升到 `0.6670`，targeted PGD worst-class robust accuracy 从 `0.1765` 提升到 `0.2621`，说明扩大训练规模后鲁棒公平性指标有明显改善。
2. 与第一周固定 target 的 targeted PGD 训练相比，本周版本不再使用固定目标类别，而是根据 false positive prior 动态采样 target class，更接近 FAIR-TAT 的方法思想。
3. `adv_weight=0.7` 相比默认 `0.5` 更偏向对抗损失，可以提高 targeted PGD original accuracy 和 worst-class robust accuracy。
4. `prior_smoothing=10.0` 比较适合当前 SmallCNN 设置，能够让 target prior 更平滑，训练表现更稳定。
5. 当前最佳结果为 `stage08a_adv07_smoothing10_full`，targeted PGD worst-class robust accuracy 达到 `0.2961`，相比 full 基线 `0.2621` 有进一步提升。
6. `stage08b` 的 targeted PGD original accuracy 和 FGSM worst-class robust accuracy 更高，但 targeted PGD worst-class robust accuracy 低于 `stage08a`，因此如果优先关注 FAIR-TAT 的最差类鲁棒性，当前应选择 `stage08a` 作为最佳模型。

## 五、当前问题

1. 当前实现仍是 FAIR-TAT-like 轻量复现，还未完全实现原文中的完整动态训练策略。
2. 当前模型仍为 SmallCNN，后续需要尝试更强 backbone，例如 ResNet 系列。
3. CIFAR-10 只有类别标签，没有医疗场景中的敏感属性，公平性分析目前主要是 class-wise fairness。
4. 当前 FGSM worst-class robust accuracy 仍然偏低，说明 targeted PGD 训练不能完全覆盖不同攻击类型下的最差类鲁棒性。
5. ISIC 2019 等医疗图像数据集还需要继续整理标签、元数据属性和公平性评测设置。

## 六、下周计划

1. 继续对齐 FAIR-TAT 原文，补充更完整的 target class 更新和 per-class 鲁棒性调节策略。
2. 在当前最佳参数 `adv_weight=0.7, prior_smoothing=10.0, PGD-5` 上进一步复现实验，确认结果稳定性。
3. 尝试更强模型结构，优先考虑 ResNet18 或轻量 ResNet。
4. 增加更强评测攻击，例如 untargeted PGD、多步 FGSM 或 AutoAttack 的轻量替代版本。
5. 开始调研 ISIC 2019 数据集，整理疾病标签、年龄、性别、解剖部位等属性，并设计医疗图像公平性评测方案。

## 七、材料位置
实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：

- FAIR-TAT-like 方法说明：`week2/docs/fair_tat_like_cifar10_notes.md`
- 阶段实验结果总表：`week2/docs/fair_tat_like_stage02_results.md`
- 后续实验计划：`week2/docs/week2_next_experiments.md`
- 本周周报：`week2/docs/weekly_report.md`
- 主要实验目录：`week2/experiments/stage01_fair_tat_like_smoke` 至 `week2/experiments/stage08b_adv07_smoothing10_pgd7_full`
- 当前最佳模型结果：`week2/experiments/stage08a_adv07_smoothing10_full`
