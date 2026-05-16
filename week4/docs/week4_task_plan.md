# 第四周任务规划


## 一、本周核心目标

第三周实验表明，当前 FAIR-TAT-like 方法在 targeted PGD 评测下已有一定稳定性，但在更强的 untargeted PGD 评测下，worst-class robust accuracy 仍然很低。

因此第四周不只继续扩大训练规模，而是围绕一个核心问题推进：

**如何通过 per-class 鲁棒性调节，改善最差类别的鲁棒性表现。**

## 二、本周任务主线

### 任务 1：补充 per-class 鲁棒性调节策略

在现有 `false_positive_prior` 动态 target class 选择基础上，增加对类别鲁棒性缺口的利用。

计划尝试两类策略：

1. **robust-deficit target prior**
   - 统计各类别 robust accuracy。
   - 对鲁棒性较差的类别提高采样权重。
   - 与原有 false positive prior 融合，形成新的 target prior。

2. **class-weighted adversarial loss**
   - 根据每个类别的鲁棒性缺口设置 loss weight。
   - 对 worst-class 或低鲁棒类别给予更高训练权重。
   - 观察是否能提升 worst-class robust accuracy。

### 任务 2：完成 per-class 调节消融实验

先在 CIFAR-10 20k 训练集上进行消融，不直接上 full，避免单次实验成本过高。

| 阶段 | 实验名称 | 目的 |
|---|---|---|
| stage01 | best baseline 20k | 使用当前最佳参数作为对照 |
| stage02 | robust-deficit target prior | 验证 target prior 调节是否有效 |
| stage03 | class-weighted adv loss | 验证 loss weight 调节是否有效 |
| stage04 | prior + loss weight | 验证两种策略组合是否进一步提升 |

重点观察指标：

- clean accuracy
- clean worst-class accuracy
- targeted PGD worst-class robust accuracy
- untargeted PGD worst-class robust accuracy
- class-wise robust accuracy
- target prior 更新轨迹

### 任务 3：扩大 ResNet18 训练规模

第三周只完成了 ResNet18 5k / 2 epoch probe，不能判断更强模型是否真正有效。本周继续扩大 ResNet18 实验规模。

建议顺序：

1. `ResNet18 + CIFAR-10 20k + 4~6 epoch`
2. 若结果明显优于 SmallCNN，再考虑 `ResNet18 + full CIFAR-10`
3. 若结果一般，则暂缓 full ResNet18，优先回到 per-class 调节策略

重点不是单纯追求 clean accuracy，而是看：

- targeted PGD worst-class robust accuracy 是否提高。
- untargeted PGD worst-class robust accuracy 是否提高。
- class-wise 指标是否更均衡。

### 任务 4：统一强评测协议与结果汇总

为了后续实验可比较，本周固定统一评测协议：

- FGSM
- multi-step FGSM
- untargeted PGD-10
- targeted PGD
- class-wise robust accuracy
- worst-class robust accuracy

同时补充一个结果汇总表，将多个实验目录中的 `metrics.json` 整理成统一 Markdown 表格，方便写周报。

### 任务 5：推进 ISIC 2019 数据准备方案

本周医疗图像方向以准备为主，不急于正式训练。

需要整理：

- ISIC 2019 疾病类别标签。
- metadata 字段：sex、age、anatomic site。
- patient-level split 方案，避免患者泄漏。
- 医疗图像公平性评测设置：
  - disease class-wise fairness
  - sex group fairness
  - age group fairness
  - anatomical site group fairness

目标是形成一份可执行的数据处理与评测方案，为后续迁移到医疗图像分类做准备。

## 三、推荐实验顺序

| 优先级 | 实验 / 工作 | 说明 |
|---|---|---|
| P0 | 实现 per-class 调节代码 | 本周主线，优先级最高 |
| P0 | best baseline 20k 复跑 | 给消融实验提供同设置对照 |
| P0 | robust-deficit target prior 20k | 验证 prior 调节 |
| P1 | class-weighted adv loss 20k | 验证 loss 调节 |
| P1 | prior + loss weight 20k | 验证组合效果 |
| P1 | ResNet18 20k 实验 | 检查更强模型是否有效 |
| P2 | full CIFAR-10 扩大实验 | 只在 20k 结果较好时进行 |
| P2 | ISIC 2019 数据方案文档 | 为后续医疗实验铺路 |

## 四、本周预期产出

1. 一个支持 per-class robust deficit 调节的训练版本。
2. 3~4 组 CIFAR-10 20k 消融实验结果。
3. 至少一个 ResNet18 20k 实验结果。
4. 一份统一的实验结果汇总表。
5. 一份 ISIC 2019 数据处理与公平性评测方案。
6. 第四周周报中能够明确回答：
   - 当前 worst-class 鲁棒性为什么低？
   - per-class 调节是否改善了它？
   - 更强模型是否带来了额外收益？

## 五、风险与备选方案

| 风险 | 影响 | 备选方案 |
|---|---|---|
| per-class 调节导致 clean accuracy 明显下降 | 模型整体性能变差 | 降低调节权重，先只融合少量 robust deficit |
| ResNet18 CPU 训练耗时过长 | 实验进度变慢 | 先跑 20k / 4 epoch，不直接 full |
| untargeted PGD worst-class 仍然很低 | 改进不明显 | 将目标转为分析失败类别，并设计下一轮策略 |
| ISIC 数据准备耗时较长 | 医疗方向进度不足 | 本周只完成字段和评测方案，不强行训练 |

## 六、阶段判断标准

如果 per-class 调节实验满足以下任一条件，就值得继续放大到 full CIFAR-10：

- targeted PGD worst-class robust accuracy 高于当前 best baseline。
- untargeted PGD worst-class robust accuracy 有明显提升。
- class-wise robust accuracy 更均衡，最差类别不再接近 0。

如果 20k 消融没有改善，则本周周报应诚实总结为：

**当前简单 per-class 调节还不足以解决强攻击下 worst-class 鲁棒性问题，后续需要更细粒度的 target prior 更新或更强训练攻击。**
