# 周报：per-class 鲁棒调节与 ResNet18 full 实验

姓名：郭家棋

## 一、本周完成工作

本周围绕第三周发现的 worst-class robust accuracy 偏低问题，重点加入 per-class 鲁棒性调节，并将实验从 SmallCNN 扩展到 ResNet18 和 full CIFAR-10。

1. 实现 robust-deficit target prior，将低鲁棒类别纳入 target class 更新。
2. 实现 class-weighted adversarial loss，对低鲁棒类别提高训练权重。
3. 完成 SmallCNN 20k 消融实验，验证 per-class 调节有效。
4. 使用 ResNet18 进行 20k 与 full CIFAR-10 实验。
5. 使用百度 AI Studio 云端 GPU 训练，提高大规模实验效率。
6. 尝试 AMP、大 batch、PGD-5 等加速或增强设置，并分析其影响。
7. 对当前最优参数 `adv_weight=0.8` 做 seed 复验，确认结果稳定。

## 二、主要实验结果

主要设置：

- 数据集：CIFAR-10
- 模型：ResNet18
- 训练攻击：targeted PGD
- 评测攻击：FGSM、multi-step FGSM、untargeted PGD、targeted PGD
- 主要平台：百度 AI Studio，Tesla V100 GPU

| 实验 | 设置 | clean acc | clean worst | untargeted PGD worst | targeted PGD original | targeted PGD worst |
|---|---|---:|---:|---:|---:|---:|
| stage05 | ResNet18 20k baseline | 0.6407 | 0.2052 | 0.0000 | 0.4844 | 0.1165 |
| stage06 | ResNet18 20k + per-class | 0.6417 | 0.4306 | 0.0889 | 0.5234 | 0.3364 |
| stage09 | ResNet18 full + per-class, adv0.7 | 0.8035 | 0.6170 | 0.0706 | 0.6338 | 0.4261 |
| stage13 | ResNet18 full + per-class, adv0.8 | 0.8004 | **0.6390** | 0.0754 | 0.6423 | 0.4586 |
| stage14 | stage13 换 seed | **0.8012** | 0.6160 | **0.0827** | **0.6580** | **0.4687** |

## 三、结果分析

1. ResNet18 baseline 平均表现较好，但 worst-class 指标较弱，说明更强模型不能自动解决鲁棒公平性问题。
2. per-class 调节明显改善最差类鲁棒性，stage06 相比 stage05 的 targeted PGD worst-class 从 `0.1165` 提升到 `0.3364`。
3. full CIFAR-10 训练显著提高整体效果，stage09 clean accuracy 达到 `0.8035`。
4. `adv_weight=0.8` 进一步提升 worst-class 指标，stage13/stage14 是当前最佳结果。
5. AMP 与大 batch 提高训练速度，但会降低 clean accuracy 和 worst-class 表现，更适合快速探索，不适合作为最终主结果。

## 四、当前最佳结论

当前较优组合为：

`ResNet18 + full CIFAR-10 + blended target prior + class-weighted adversarial loss + adv_weight=0.8`

该组合在两个随机种子下都保持较好表现，说明结果相对稳定。

## 五、下周计划

1. 整理 per-class 调节方法说明，明确与 FAIR-TAT 原文的对应关系和差异。
2. 继续加强评测，可考虑对最优模型做更强 PGD 评测。
3. 开始准备 ISIC 2019 医疗图像数据处理流程。
4. 将当前 class-wise fairness 评测迁移到医疗图像的疾病类别、性别、年龄和解剖部位分组。

## 六、材料位置

实验材料已整理到[项目](https://github.com/Tamako521/fair_tat_project.git)目录中：
- 本周周报：`week4/docs/weekly_report.md`
- 主要实验目录：`week4/experiments/stage01_best_baseline_20k` 至 `week4/experiments/stage14_resnet18_full_adv08_pgd3_bs128_seed7`
- 当前最佳实验：`week4/experiments/stage13_resnet18_full_adv08_pgd3_bs128` 和 `week4/experiments/stage14_resnet18_full_adv08_pgd3_bs128_seed7`
