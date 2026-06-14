# 周报：ISIC 2019 Baseline 与分组公平性评测

姓名：郭家棋

## 一、本周完成工作

本周从 ISIC 2019 数据准备推进到医疗图像分类 baseline 实验，重点完成 clean baseline、迁移学习对比和分组公平性评测。

1. 修正 ISIC 2019 图片路径检查逻辑，兼容 `_downsampled` 图片命名。
2. 编写并跑通 `train_isic2019_baseline.py`，支持 ResNet18、ImageNet 预训练、AMP、类别权重和分组评测。
3. 完成 ISIC 2019 smoke test、小规模 probe、全量 baseline 和学习率优化实验。
4. 增加采样消融实验：解剖部位均衡采样、疾病类别均衡采样。
5. 输出按疾病类别、性别、年龄段和解剖部位统计的 worst-group / macro-group 指标。

## 二、主要实验设置

- 数据集：ISIC 2019
- 模型：ResNet18
- 训练方式：clean baseline
- 主要优化：ImageNet 预训练、类别权重、学习率调整、均衡采样
- 评测指标：accuracy、balanced accuracy、worst-class accuracy、分组 worst accuracy

## 三、主要实验结果

| 实验 | 设置 | test acc | test balanced acc | test worst-class | anatom-site worst |
|---|---|---:|---:|---:|---:|
| stage03 | 随机初始化 / 5k probe | 0.3370 | 0.2646 | 0.0000 | - |
| stage04 | ImageNet 预训练 / 5k probe | 0.3220 | 0.2823 | 0.1765 | - |
| stage05 | 预训练 / full baseline / lr=1e-4 | 0.6436 | 0.5617 | 0.2796 | 0.0000 |
| stage06 | 预训练 / full / lr=5e-5 | 0.6587 | **0.5850** | **0.3011** | **0.2000** |
| stage07 | 解剖部位均衡采样 | **0.6942** | 0.5712 | 0.2258 | 0.0000 |
| stage08 | 疾病类别均衡采样 | 0.5317 | 0.5672 | 0.2366 | 0.1000 |

## 四、结果分析

1. ImageNet 预训练能明显改善最差类别表现，小规模实验中 worst-class 从 0 提升到 0.1765。
2. 全量数据训练后，模型性能明显提升，说明 ISIC 2019 baseline 流程有效。
3. stage06 是当前综合最优结果，test balanced accuracy 和 worst-class accuracy 均最高。
4. 解剖部位均衡采样提升了整体 accuracy、性别组和年龄组表现，但没有稳定改善最差解剖部位。
5. 疾病类别均衡采样保留了一定 balanced accuracy，但整体 accuracy 下降明显，说明简单采样存在精度和公平性的权衡。

## 五、本周结论

本周已经完成 ISIC 2019 clean baseline 的完整实验链条。当前建议将 stage06 作为主结果，stage07 和 stage08 作为公平性采样消融实验。结果表明，医疗图像数据上的类别不平衡和属性分组不平衡较明显，后续需要更细粒度的 fairness-aware loss 或对抗训练策略。

## 六、下周计划

1. 在 ISIC 2019 baseline 上加入更细的分组损失或 worst-group reweighting。
2. 尝试将 FAIR-TAT-like targeted PGD 迁移到 ISIC 2019 小规模实验。
3. 重点观察对抗训练对 worst-class accuracy 和分组 worst accuracy 的影响。
4. 继续保留 stage06 作为 clean baseline 对照。

## 七、材料位置

实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：
- ISIC baseline 脚本：`scripts/train_isic2019_baseline.py`
- 主要实验：`week7/experiments/stage02_isic2019_baseline_smoke` 至 `week7/experiments/stage08_isic2019_pretrained_full_label_sampler`
- 本周周报：`week7/docs/weekly_report.md`
