# ISIC 2019 公平性评测方案

姓名：郭家棋

## 一、目标

本方案用于将当前 CIFAR-10 / CIFAR-100 中的 class-wise fairness 评测迁移到医疗图像分类任务。ISIC 2019 不仅可以按疾病类别评测模型表现，也可以按性别、年龄和解剖部位等属性分析公平性。

## 二、评测维度

| 维度 | 分组方式 | 关注点 |
|---|---|---|
| 疾病类别 | MEL、NV、BCC、AK、BKL、DF、VASC、SCC | 类别间准确率差异 |
| 性别 | male、female、unknown | 性别分组表现差异 |
| 年龄 | 0-30、31-45、46-60、61+、unknown | 年龄段表现差异 |
| 解剖部位 | head/neck、torso、upper extremity、lower extremity、palms/soles、oral/genital、unknown | 不同部位的表现差异 |
| 患者级别 | patient-level split | 避免同一患者泄漏到训练和测试 |

## 三、核心指标

基础分类指标：

- overall accuracy
- balanced accuracy
- per-class accuracy
- worst-class accuracy

公平性指标：

- per-group accuracy
- worst-group accuracy
- group accuracy gap
- macro average group accuracy

鲁棒性指标：

- clean accuracy
- adversarial accuracy
- per-class robust accuracy
- worst-class robust accuracy
- per-group robust accuracy
- worst-group robust accuracy

## 四、初步实验流程

1. 读取 ISIC 2019 图像、标签和 metadata。
2. 统计疾病类别、性别、年龄、解剖部位的样本数量和缺失比例。
3. 按 patient_id 做训练集 / 验证集划分。
4. 先训练 clean baseline，记录疾病类别和属性分组表现。
5. 再迁移 targeted PGD / FAIR-TAT-like 训练入口。
6. 比较 clean baseline 与 adversarial training 在不同分组上的表现。

## 五、与当前 FAIR-TAT-like 项目的对应关系

当前 CIFAR 实验中：

- 类别 = CIFAR 类别
- fairness group = class
- 主要指标 = class-wise accuracy / worst-class robust accuracy

迁移到 ISIC 2019 后：

- 类别 = 疾病诊断类别
- fairness group = 疾病类别 + 性别 + 年龄段 + 解剖部位
- 主要指标 = worst-class accuracy + worst-group accuracy

## 六、下一步实现建议

优先新增一个数据分析脚本：

`scripts/analyze_isic2019_metadata.py`

建议输出：

- 标签分布表
- 性别分布表
- 年龄段分布表
- 解剖部位分布表
- 缺失值统计
- patient_id / lesion_id 重复情况

后续再新增：

`scripts/train_isic2019_baseline.py`

用于先跑 clean baseline，再考虑接入对抗训练。

当前已新增 metadata 分析脚本，运行方式如下：

```bash
python scripts/analyze_isic2019_metadata.py \
  --metadata data/ISIC_2019_Training_Metadata.csv \
  --output-dir week6/experiments/isic2019_metadata_analysis
```

脚本输出：

- `isic2019_metadata_summary.json`
- `isic2019_metadata_summary.md`

## 七、阶段性结论

第六周 ISIC 2019 方向先完成数据字段和公平性评测方案设计即可。等 metadata 统计完成后，再决定是否进行小规模图像训练。
