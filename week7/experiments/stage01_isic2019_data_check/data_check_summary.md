# ISIC 2019 数据可用性检查

## 一、总体结果

- splits_dir：`/workspace/week6/experiments/isic2019_splits`
- image_dir：`/workspace/data/isic2019/images`
- 总样本数：25331
- 图片存在数：0
- 图片缺失数：25331

## train

- 样本数：17740
- 图片存在数：0
- 图片缺失数：17740
- lesion_id 数：8526
- 缺失字段：[]

### 标签分布

| 标签 | 数量 |
|---|---:|
| NV | 9014 |
| MEL | 3166 |
| BCC | 2328 |
| BKL | 1838 |
| AK | 607 |
| SCC | 440 |
| VASC | 179 |
| DF | 168 |

### 性别分布

| 性别 | 数量 |
|---|---:|
| male | 9325 |
| female | 8133 |
| unknown | 282 |

### 年龄段分布

| 年龄段 | 数量 |
|---|---:|
| 61+ | 6186 |
| 46-60 | 4623 |
| 31-45 | 4578 |
| 0-30 | 2005 |
| unknown | 348 |

### 解剖部位分布

| 部位 | 数量 |
|---|---:|
| anterior torso | 4846 |
| lower extremity | 3571 |
| head/neck | 3259 |
| upper extremity | 2030 |
| posterior torso | 1877 |
| unknown | 1799 |
| palms/soles | 252 |
| oral/genital | 64 |
| lateral torso | 42 |

### 图片抽样打开检查

- 启用：True
- PIL 可用：True
- 成功打开数量：0
- 打开失败数量：0

### 缺失图片示例

```text
ISIC_0031073
ISIC_0027477
ISIC_0028057
ISIC_0031502
ISIC_0027971
ISIC_0029621
ISIC_0033121
ISIC_0033213
ISIC_0034158
ISIC_0024538
ISIC_0025208
ISIC_0029499
ISIC_0033647
ISIC_0030376
ISIC_0056078
ISIC_0056260
ISIC_0068337
ISIC_0069134
ISIC_0054631
ISIC_0032398
```

## val

- 样本数：3820
- 图片存在数：0
- 图片缺失数：3820
- lesion_id 数：1884
- 缺失字段：[]

### 标签分布

| 标签 | 数量 |
|---|---:|
| NV | 1934 |
| MEL | 688 |
| BCC | 501 |
| BKL | 396 |
| AK | 132 |
| SCC | 95 |
| VASC | 38 |
| DF | 36 |

### 性别分布

| 性别 | 数量 |
|---|---:|
| male | 1908 |
| female | 1853 |
| unknown | 59 |

### 年龄段分布

| 年龄段 | 数量 |
|---|---:|
| 61+ | 1325 |
| 46-60 | 987 |
| 31-45 | 979 |
| 0-30 | 451 |
| unknown | 78 |

### 解剖部位分布

| 部位 | 数量 |
|---|---:|
| anterior torso | 1015 |
| lower extremity | 714 |
| head/neck | 671 |
| upper extremity | 467 |
| posterior torso | 448 |
| unknown | 399 |
| palms/soles | 68 |
| oral/genital | 29 |
| lateral torso | 9 |

### 图片抽样打开检查

- 启用：True
- PIL 可用：True
- 成功打开数量：0
- 打开失败数量：0

### 缺失图片示例

```text
ISIC_0032998
ISIC_0033633
ISIC_0034199
ISIC_0025047
ISIC_0061837
ISIC_0069817
ISIC_0026225
ISIC_0013342
ISIC_0025270
ISIC_0029218
ISIC_0031576
ISIC_0059097
ISIC_0059404
ISIC_0069101
ISIC_0027871
ISIC_0055939
ISIC_0058585
ISIC_0055428
ISIC_0068706
ISIC_0053975
```

## test

- 样本数：3771
- 图片存在数：0
- 图片缺失数：3771
- lesion_id 数：1854
- 缺失字段：[]

### 标签分布

| 标签 | 数量 |
|---|---:|
| NV | 1927 |
| MEL | 668 |
| BCC | 494 |
| BKL | 390 |
| AK | 128 |
| SCC | 93 |
| VASC | 36 |
| DF | 35 |

### 性别分布

| 性别 | 数量 |
|---|---:|
| male | 2053 |
| female | 1675 |
| unknown | 43 |

### 年龄段分布

| 年龄段 | 数量 |
|---|---:|
| 61+ | 1258 |
| 46-60 | 1085 |
| 31-45 | 925 |
| 0-30 | 447 |
| unknown | 56 |

### 解剖部位分布

| 部位 | 数量 |
|---|---:|
| anterior torso | 1054 |
| lower extremity | 715 |
| head/neck | 657 |
| posterior torso | 462 |
| upper extremity | 413 |
| unknown | 384 |
| palms/soles | 68 |
| oral/genital | 15 |
| lateral torso | 3 |

### 图片抽样打开检查

- 启用：True
- PIL 可用：True
- 成功打开数量：0
- 打开失败数量：0

### 缺失图片示例

```text
ISIC_0000092
ISIC_0024737
ISIC_0026260
ISIC_0029869
ISIC_0032399
ISIC_0000250
ISIC_0025001
ISIC_0028800
ISIC_0061325
ISIC_0072019
ISIC_0025007
ISIC_0028661
ISIC_0025787
ISIC_0028618
ISIC_0031488
ISIC_0026066
ISIC_0016055
ISIC_0011084
ISIC_0027418
ISIC_0029030
```
