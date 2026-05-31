# ISIC 2019 数据字段整理

姓名：郭家棋

## 一、数据集概况

ISIC 2019 是皮肤病变图像分类数据集，可用于医疗图像分类与公平性评测。本项目后续计划将 CIFAR-10 / CIFAR-100 上的 class-wise fairness 评测思路迁移到 ISIC 2019。

官方数据规模：

- 训练图像：33,126 张 JPEG 图像
- 训练标签：33,126 条 gold standard lesion diagnosis
- metadata：patient ID、lesion ID、sex、age、general anatomic site
- 测试图像：10,982 张 JPEG 图像

## 二、疾病类别

ISIC 2019 分类任务主要包含 8 个皮肤病变类别：

| 缩写 | 类别含义 |
|---|---|
| MEL | melanoma，黑色素瘤 |
| NV | melanocytic nevus，黑色素细胞痣 |
| BCC | basal cell carcinoma，基底细胞癌 |
| AK | actinic keratosis，光化性角化病 |
| BKL | benign keratosis-like lesions，良性角化样病变 |
| DF | dermatofibroma，皮肤纤维瘤 |
| VASC | vascular lesions，血管性病变 |
| SCC | squamous cell carcinoma，鳞状细胞癌 |

说明：ISIC 2019 测试阶段还涉及 unknown / outlier 类，但训练集主要按上述 8 类进行建模。

## 三、metadata 字段

后续重点关注以下字段：

| 字段 | 含义 | 用途 |
|---|---|---|
| image / image_id | 图像编号 | 与图像文件和标签表关联 |
| patient_id | 患者编号 | 用于 patient-level split，避免同一患者泄漏到训练和测试 |
| lesion_id | 病灶编号 | 用于识别同一病灶的多张图像 |
| sex | 性别 | 公平性分组属性 |
| age_approx / age | 近似年龄 | 公平性分组属性 |
| anatom_site_general | 解剖部位 | 公平性分组属性 |
| diagnosis / label | 疾病类别 | 分类目标 |

## 四、需要注意的问题

1. 类别不均衡明显，不能只看整体 accuracy。
2. 年龄、性别、解剖部位可能存在缺失值，需要单独统计。
3. 同一患者或同一病灶可能对应多张图像，划分数据集时应优先考虑 patient-level split。
4. 医疗图像评测更适合使用 balanced accuracy、per-class accuracy 和 worst-group accuracy。
5. 后续若引入对抗训练，需要同时观察疾病类别公平性和属性分组公平性。

## 五、参考来源

- ISIC 2019 Challenge：https://challenge.isic-archive.com/landing/2019/
- ISIC Challenge Data：https://challenge.isic-archive.com/data/
- Gessert et al., Skin Lesion Classification Using Ensembles of Multi-Resolution EfficientNets with Meta Data：https://arxiv.org/abs/1910.03910
