# 第六周任务计划

姓名：郭家棋

## 一、本周目标

第六周采用均衡推进方案：一方面继续优化 CIFAR-100 上的 FAIR-TAT-like 实验，重点改善 worst-class robust accuracy；另一方面开始准备 ISIC 2019 医疗图像数据集，为后续医疗图像公平性实验做铺垫。

任务比例：

- CIFAR-100 优化实验：60%
- ISIC 2019 调研与数据准备：40%

## 二、CIFAR-100 实验任务

第五周结果表明，CIFAR-100 上 clean accuracy 能随数据规模提升，但 worst-class robust accuracy 仍然偏低。因此本周不继续盲目增加训练时长，而是做有针对性的参数实验。

计划实验：

| 阶段 | 实验名称 | 目的 |
|---|---|---|
| stage01 | `cifar100_adv_weight_07` | 降低 `adv_weight`，观察 clean 与 robust 的平衡 |
| stage02 | `cifar100_deficit_weight_09` | 提高 `deficit_weight`，增强低鲁棒类别在 target prior 中的影响 |
| stage03 | `cifar100_loss_weight_15` | 使用中等 class weight，避免 stage04 权重过强导致 clean 下降 |
| stage04 | `cifar100_best_fast_probe` | 从前三个实验中选较优配置，做 full fast probe |

推荐基础设置：

- 数据集：CIFAR-100
- 模型：ResNet18
- 训练规模：优先 20k / 4 epochs
- 训练攻击：targeted PGD
- 评测攻击：FGSM、multi-step FGSM、untargeted PGD、targeted PGD
- 稳定配置：不使用 AMP，必要时使用 `--disable-cudnn`

重点观察指标：

- clean accuracy
- clean worst-class accuracy
- untargeted PGD robust accuracy
- untargeted PGD worst-class accuracy
- targeted PGD original accuracy
- targeted PGD worst-class accuracy

## 三、ISIC 2019 准备任务

本周 ISIC 2019 以调研和数据准备为主，不强求完成训练。

需要整理：

1. ISIC 2019 的疾病类别标签。
2. 可用公平性属性：年龄、性别、解剖部位。
3. 数据文件结构和 metadata 字段。
4. 医疗图像公平性评测方案。

初步评测方案：

| 评测维度 | 说明 |
|---|---|
| 疾病类别 | 统计每个疾病类别的分类准确率和 worst-class accuracy |
| 性别分组 | 比较 male / female / unknown 分组表现 |
| 年龄分组 | 按年龄段统计模型表现 |
| 解剖部位 | 比较不同 anatomical site 的分类差异 |

## 四、本周预期产出

1. 2-3 个 CIFAR-100 参数优化实验结果。
2. 一个 CIFAR-100 较优配置的 full fast probe。
3. 一份 ISIC 2019 字段整理笔记。
4. 一份医疗图像公平性评测设计说明。
5. 第六周周报中形成“CIFAR-100 优化 + 医疗数据准备”的双线成果。

## 五、当前约束

1. AI Studio 环境中 AMP / cuDNN 存在不稳定情况，训练优先采用稳定配置。
2. CIFAR-100 训练耗时明显高于 CIFAR-10，本周避免长时间 full 训练。
3. ISIC 2019 若数据下载耗时较长，本周优先完成 metadata 和评测方案。

## 六、Decision Log

- 已确认：第六周采用方案 A，均衡推进。
- 已确认：CIFAR-100 占 60%，ISIC 2019 占 40%。
- 已确认：CIFAR-100 本周重点改善 worst-class robust accuracy。
- 已确认：ISIC 2019 本周以调研、字段整理和评测方案为主。
