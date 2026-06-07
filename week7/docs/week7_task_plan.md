# 第七周任务计划

姓名：郭家棋

## 一、本周目标

第七周从“数据准备”推进到“医疗图像 baseline 实验”。主线是基于第六周完成的 ISIC 2019 metadata 统计和 train / val / test 划分，完成 ISIC 2019 clean baseline，并实现基础分组公平性评测。

任务比例：

- ISIC 2019 clean baseline 与公平性评测：70%
- CIFAR-100 worst-class 后续分析：30%

## 二、ISIC 2019 任务

本周重点不急于上对抗训练，先跑通医疗图像分类 baseline。

计划任务：

| 阶段 | 任务 | 目的 |
|---|---|---|
| stage01 | 检查 ISIC 图像路径与 split CSV | 确认训练数据可读取 |
| stage02 | 编写 `train_isic2019_baseline.py` | 完成 clean baseline 训练入口 |
| stage03 | 跑小规模 smoke test | 验证模型、数据加载和日志保存 |
| stage04 | 跑 ResNet18 clean baseline | 获得第一个医疗图像分类结果 |
| stage05 | 实现分组评测 | 按疾病类别、性别、年龄段、解剖部位统计表现 |

stage01 运行命令：

```bash
python scripts/check_isic2019_data.py \
  --splits-dir week6/experiments/isic2019_splits \
  --image-dir data/isic2019/images \
  --output-dir week7/experiments/stage01_isic2019_data_check \
  --max-open-per-split 32
```

该脚本会检查：

- train / val / test CSV 是否存在
- 必要字段是否完整
- 图片文件是否存在
- 每个 split 的疾病类别、性别、年龄段、解剖部位分布
- 每个 split 抽样打开图片是否成功

重点指标：

- overall accuracy
- balanced accuracy
- per-class accuracy
- worst-class accuracy
- sex-group accuracy
- age-group accuracy
- anatom-site group accuracy
- worst-group accuracy

## 三、CIFAR-100 任务

第六周结果显示 `deficit_weight=0.9` 能提升平均表现，但 targeted PGD worst-class 仍为 0。本周不继续大量训练，只做分析和轻量验证。

计划任务：

1. 汇总 week5 / week6 CIFAR-100 实验结果。
2. 分析最差类长期为 0 的原因。
3. 设计一个轻量策略：延长 clean warmup 或降低训练攻击强度，观察 worst-class 是否改善。
4. 若时间允许，跑一个 20k 小实验。

推荐轻量实验：

`stage01_cifar100_longer_warmup_20k`

目的：验证更长 clean warmup 是否能改善 CIFAR-100 上低频类别学习不足的问题。

## 四、本周预期产出

1. ISIC 2019 clean baseline 训练脚本。
2. ISIC 2019 smoke test 结果。
3. ISIC 2019 clean baseline 结果。
4. ISIC 2019 分组公平性评测结果。
5. CIFAR-100 worst-class 问题分析或一个轻量补充实验。

## 五、当前约束

1. 腾讯 CNB 工作区数据不能持久保存，数据和模型需要推送到对象存储。
2. 大文件应通过 `.gitattributes` / LFS 规则跟踪。
3. ISIC 图像数据较大，训练前需要确认图片是否已在当前工作区或对象存储中恢复。
4. 第七周优先保证 clean baseline 和分组评测跑通，不急于迁移 FAIR-TAT-like 对抗训练。

## 六、实验顺序

1. 检查 ISIC 图片与 split 文件。
2. 写并运行 ISIC baseline smoke test。
3. 运行 ISIC clean baseline。
4. 输出分组评测表。
5. 根据时间决定是否补一个 CIFAR-100 longer warmup 实验。

## 七、Decision Log

- 已确认：第七周主线从 CIFAR-100 转向 ISIC 2019 clean baseline。
- 已确认：ISIC 2019 优先做 clean baseline 与公平性分组评测。
- 已确认：CIFAR-100 只做轻量后续分析，不再作为本周主训练任务。
