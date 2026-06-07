# 周报：CIFAR-100 优化与 ISIC 2019 数据准备

姓名：郭家棋

## 一、本周完成工作

本周采用双线推进：一方面继续优化 CIFAR-100 上的 FAIR-TAT-like 实验，另一方面开始准备 ISIC 2019 医疗图像数据集。

1. 完成 4 组 CIFAR-100 参数实验，重点观察 `adv_weight`、`deficit_weight` 和 `loss_weight_strength` 对鲁棒性的影响。
2. 在 full fast probe 中验证 `deficit_weight=0.9` 的效果。
3. 完成 ISIC 2019 metadata 统计，整理疾病类别、性别、年龄和解剖部位分布。
4. 实现 ISIC 2019 数据划分脚本，生成 train / val / test CSV。
5. 初步完成医疗图像公平性评测方案设计。

## 二、CIFAR-100 实验结果

主要设置：

- 数据集：CIFAR-100
- 模型：ResNet18
- 训练攻击：targeted PGD
- 平台：百度 AI Studio，Tesla V100 GPU

| 实验 | 设置 | clean acc | clean worst | untargeted PGD | targeted PGD original | targeted PGD worst |
|---|---|---:|---:|---:|---:|---:|
| week5 full probe | full / `deficit_weight=0.5` | 0.3908 | 0.0179 | 0.0898 | 0.3516 | 0.0000 |
| stage01 | 20k / `adv_weight=0.7` | 0.3367 | 0.0000 | 0.0801 | 0.2725 | 0.0000 |
| stage02 | 20k / `deficit_weight=0.9` | 0.3267 | 0.0000 | 0.0879 | 0.2773 | 0.0000 |
| stage03 | 20k / `loss_weight_strength=1.5` | 0.3270 | 0.0000 | 0.0859 | 0.2754 | 0.0000 |
| stage04 | full probe / `deficit_weight=0.9` | **0.4310** | 0.0164 | **0.1055** | **0.4023** | 0.0000 |

## 三、CIFAR-100 结果分析

1. 降低 `adv_weight` 到 0.7 后，clean accuracy 略升，但鲁棒性提升不明显。
2. 提高 `deficit_weight` 到 0.9 后，平均鲁棒性略有提升。
3. `loss_weight_strength=1.5` 没有明显优于默认设置。
4. 当前较优结果是 stage04，相比第五周 full probe，clean accuracy、untargeted PGD 和 targeted PGD original 都有提升。
5. targeted PGD worst-class 仍为 0，说明 CIFAR-100 的最差类鲁棒性仍是后续主要问题。

## 四、ISIC 2019 数据准备

已完成 metadata 统计：

- 样本数：25,331
- 性别：male 13,286，female 11,661，unknown 384
- 年龄：61+ 为 8,769，46-60 为 6,695，31-45 为 6,482，0-30 为 2,903，unknown 482
- 主要解剖部位：anterior torso、lower extremity、head/neck、upper extremity、posterior torso
- lesion_id 可用，适合用于降低同一病灶泄漏风险

已生成数据划分：

| split | 样本数 | lesion_id 数 |
|---|---:|---:|
| train | 17,740 | 8,526 |
| val | 3,820 | 1,884 |
| test | 3,771 | 1,854 |

## 五、本周结论

本周完成了 CIFAR-100 参数优化和 ISIC 2019 数据准备两条线。CIFAR-100 上 `deficit_weight=0.9` 在 full fast probe 中取得当前较优结果；ISIC 2019 已完成 metadata 分析和 train / val / test 划分，可作为后续医疗图像公平性实验基础。

## 六、下周计划

1. 继续改进 CIFAR-100 的 worst-class robust accuracy。
2. 基于 ISIC 2019 split 编写 clean baseline 训练入口。
3. 在 ISIC 2019 上实现按疾病类别、性别、年龄段和解剖部位的分组评测。
4. 评估是否将 FAIR-TAT-like 对抗训练迁移到 ISIC 2019。

## 七、材料位置
实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：
- CIFAR-100 实验：`week6/experiments/stage01_cifar100_adv_weight_07` 至 `week6/experiments/stage04_cifar100_deficit09_full_fast_probe`
- ISIC metadata 统计：`week6/experiments/isic2019_metadata_analysis`
- ISIC 数据划分：`week6/experiments/isic2019_splits`
