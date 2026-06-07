# ISIC 2019 Metadata 统计结果

## 一、基本信息

- 样本数：25331
- 字段数：35
- ground truth 合并数量：23257

## 二、疾病类别分布

| 类别 | 数量 |
|---|---:|
| NV | 11559 |
| MEL | 4148 |
| BCC | 3323 |
| BKL | 2240 |
| nevus | 1316 |
| AK | 867 |
| SCC | 628 |
| seborrheic keratosis | 384 |
| melanoma | 374 |
| VASC | 253 |
| DF | 239 |

## 三、性别分布

| 性别 | 数量 |
|---|---:|
| male | 13286 |
| female | 11661 |
| unknown | 384 |

## 四、年龄段分布

| 年龄段 | 数量 |
|---|---:|
| 61+ | 8769 |
| 46-60 | 6695 |
| 31-45 | 6482 |
| 0-30 | 2903 |
| unknown | 482 |

## 五、解剖部位分布

| 解剖部位 | 数量 |
|---|---:|
| anterior torso | 6915 |
| lower extremity | 5000 |
| head/neck | 4587 |
| upper extremity | 2910 |
| posterior torso | 2787 |
| unknown | 2582 |
| palms/soles | 388 |
| oral/genital | 108 |
| lateral torso | 54 |

## 六、患者与病灶重复情况

- patient_id 可用：True
- patient_id 唯一数量：369
- 出现多张图像的 patient_id 数：30
- 单个 patient_id 最大图像数：7
- lesion_id 可用：True
- lesion_id 唯一数量：12264
- 出现多张图像的 lesion_id 数：5059
- 单个 lesion_id 最大图像数：31

## 七、缺失值最多的字段

| 字段 | 缺失数量 |
|---|---:|
| image | 25331 |
| UNK | 25331 |
| diagnosis_5 | 24917 |
| patient_id | 24914 |
| family_hx_mm | 24786 |
| personal_hx_mm | 24776 |
| clin_size_long_diam_mm | 24775 |
| nevus_type | 24750 |
| anatom_site_special | 24371 |
| diagnosis_4 | 24355 |
| dermoscopic_type | 23901 |
| benign_malignant | 7547 |
| anatom_site_general | 2582 |
| diagnosis_confirm_type | 2459 |
| MEL | 2074 |
| NV | 2074 |
| BCC | 2074 |
| AK | 2074 |
| BKL | 2074 |
| DF | 2074 |

## 八、后续建议

1. 优先使用 patient-level split，避免同一患者进入训练集和验证集。
2. 训练和评测时同时关注疾病类别、性别、年龄段和解剖部位分组。
3. 缺失值较多的属性应保留 `unknown` 分组，不建议直接删除样本。
