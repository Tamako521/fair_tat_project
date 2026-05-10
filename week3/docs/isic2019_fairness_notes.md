# ISIC 2019 医疗图像公平性调研笔记

## 数据集概况

ISIC Challenge 官方页面显示，ISIC 2019 训练集提供：

- 33,126 张训练图像。
- JPEG 图像版本约 23GB。
- 训练 metadata 包含 patient ID、lesion ID、sex、age、general anatomic site。
- 训练 ground truth 包含 33,126 条皮肤病变诊断标签。
- 测试集提供 10,982 张图像及 metadata，但官方页面标注测试 ground truth 不可用。
- 数据许可为 CC-BY-NC。

参考来源：https://challenge.isic-archive.com/data/

## 可用标签

ISIC 2019 任务通常用于多分类皮肤病变诊断。后续下载 ground truth 后需要确认具体列名，预期可整理为：

- melanoma
- melanocytic nevus
- basal cell carcinoma
- actinic keratosis / Bowen disease
- benign keratosis
- dermatofibroma
- vascular lesion
- squamous cell carcinoma
- unknown / none of the above

实际实验中应以官方 CSV 表头为准，不手写替代标签。

## 可用公平性属性

metadata 中可直接用于公平性评测的属性：

- sex：按性别分组。
- age：按年龄段分组，例如 `<30`, `30-50`, `50-70`, `>=70`。
- general anatomic site：按解剖部位分组，例如 head/neck、torso、upper extremity、lower extremity 等。
- patient ID / lesion ID：用于避免同一患者或同一病灶泄漏到不同数据划分。

## 后续公平性评测设计

医疗图像阶段可以沿用 CIFAR-10 的两层公平性思路：

1. 类别级公平性：统计每个疾病类别的 clean accuracy、robust accuracy、worst-class robust accuracy。
2. 属性级公平性：在 sex、age group、anatomic site 分组下统计 clean accuracy、robust accuracy、worst-group robust accuracy。

推荐指标：

- overall accuracy
- balanced accuracy
- per-class accuracy
- per-class robust accuracy
- worst-class robust accuracy
- per-attribute-group accuracy
- worst-group robust accuracy
- group gap，即最好组与最差组之间的差距

## 数据划分注意事项

- 使用 patient-level split，避免同一患者图像同时出现在训练集和测试集。
- 类别极不均衡，需要记录每个类别和每个属性组样本数。
- 训练阶段可先从 resized JPEG 子集开始，不急于直接处理完整 DICOM。
- 医疗数据实验报告中要明确说明这是科研复现与算法评估，不做临床诊断结论。
