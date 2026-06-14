# 第八周 1 小时实验计划

姓名：郭家棋

## 一、本周目标

第八周不做长时间大规模训练，目标是在 1 小时内完成一个完整、可汇报的小闭环：在第七周 ISIC 2019 clean baseline 的基础上，补充一个轻量对抗评测 / 鲁棒性分析实验，观察 clean baseline 在 FGSM 或 PGD 攻击下的类别与分组表现。

核心思路：

- 保留第七周最佳 clean baseline：`stage06_isic2019_pretrained_full_lr5e5`
- 不重新训练大模型，优先复用已有 checkpoint
- 做轻量 adversarial evaluation，输出 class-wise 和 group-wise robust accuracy
- 若时间不足，先完成 FGSM；若时间充足，再补 PGD-3

## 二、1 小时任务安排

| 时间 | 任务 | 产出 |
|---|---|---|
| 0-10 分钟 | 检查第七周最佳模型与 split 文件 | 确认 checkpoint、CSV、图片路径可用 |
| 10-30 分钟 | 编写或补全 ISIC 对抗评测脚本 | 支持加载模型、FGSM/PGD 评测、分组统计 |
| 30-45 分钟 | 跑 stage01 轻量 FGSM 评测 | 生成 `metrics.json` 和日志 |
| 45-55 分钟 | 若时间允许，跑 stage02 PGD-3 小样本评测 | 作为 stronger attack 对照 |
| 55-60 分钟 | 整理结果摘要 | 得到可写入周报的表格和结论 |

## 三、推荐实验

### stage01：FGSM 鲁棒性快速评测

目的：验证 ISIC clean baseline 在简单对抗攻击下的鲁棒性，并统计最差类别和最差分组。

推荐设置：

- checkpoint：`week7/experiments/stage06_isic2019_pretrained_full_lr5e5/model.pth`
- attack：FGSM
- epsilon：`2/255` 或 `4/255`
- eval size：512 或 1024
- 关注指标：robust accuracy、worst-class robust accuracy、sex / age / anatom-site worst-group robust accuracy

### stage02：PGD-3 小样本评测

目的：补充更强攻击下的鲁棒性观察，作为 FGSM 的对照。

推荐设置：

- checkpoint 同 stage01
- attack：untargeted PGD-3
- epsilon：`2/255`
- alpha：`1/255`
- eval size：512

## 四、成功标准

1. 能加载第七周 stage06 模型完成评测。
2. 至少输出 FGSM 下的 overall robust accuracy 和 worst-class robust accuracy。
3. 至少输出一个分组维度的 worst-group robust accuracy，优先选择 `anatom_site_general`。
4. 形成一条清晰结论：clean baseline 在对抗扰动下哪些类别或属性组更脆弱。

## 五、预期周报写法

如果实验跑通，本周可汇报为：

1. 在 ISIC 2019 clean baseline 基础上补充轻量对抗评测。
2. 初步比较 clean accuracy 与 adversarial accuracy 的差异。
3. 分析疾病类别和解剖部位分组在对抗扰动下的鲁棒性变化。
4. 为后续将 FAIR-TAT-like targeted adversarial training 迁移到 ISIC 2019 提供 baseline 对照。

## 六、风险与备选方案

| 风险 | 处理方式 |
|---|---|
| PGD 评测太慢 | 只保留 FGSM 评测 |
| 显存不足 | 降低 batch size 到 64 或 32 |
| 对抗脚本未完成 | 先输出 clean baseline 的分组弱点分析 |
| checkpoint 路径不一致 | 使用 stage06 的 `model_best.pth` 或 `model.pth` 二选一 |

## 七、Decision Log

- 决定：第八周做轻量对抗评测，不做新一轮全量训练。
- 原因：1 小时时间限制下，复用已有 stage06 checkpoint 更稳。
- 备选：继续做采样训练；暂不采用，因为第七周已完成 stage07 / stage08 消融。
- 备选：直接迁移 FAIR-TAT-like 训练；暂不采用，因为医疗图像对抗训练成本较高，适合作为下一步较长实验。
