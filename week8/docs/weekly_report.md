# 周报：ISIC 2019 轻量对抗评测

姓名：郭家棋

## 一、本周完成工作

本周基于第七周表现最好的 ISIC 2019 clean baseline，补充轻量对抗评测，重点观察模型在 FGSM、PGD 和随机噪声扰动下的鲁棒性。

1. 编写 `evaluate_isic2019_adversarial.py`，支持加载已有模型进行对抗评测。
2. 实现 FGSM、PGD-3 和 random noise 三种评测方式。
3. 在 test split 上输出 overall accuracy、worst-class accuracy 和分组 worst accuracy。
4. 增加 `epsilon=0` sanity check，验证评测脚本结果可信。

## 二、实验设置

- 数据集：ISIC 2019
- 模型：第七周最佳 ResNet18 clean baseline
- checkpoint：`week7/experiments/stage06_isic2019_pretrained_full_lr5e5/model.pth`
- 评测样本：FGSM / random / sanity 为 1024，PGD-3 为 512
- 平台：腾讯 CNB 云显卡
- 主要指标：adversarial accuracy、worst-class robust accuracy、anatom-site worst-group accuracy

## 三、实验结果

| 实验 | 扰动方式 | epsilon | clean acc | adv acc | worst-class |
|---|---|---:|---:|---:|---:|
| stage06 | FGSM sanity | 0 | 0.6436 | 0.6436 | 0.2308 |
| stage05 | FGSM | 0.5/255 | 0.6436 | 0.0342 | 0.0000 |
| stage02 | FGSM | 1/255 | 0.6436 | 0.0303 | 0.0000 |
| stage01 | FGSM | 2/255 | 0.6436 | 0.0381 | 0.0000 |
| stage04 | PGD-3 | 1/255 | 0.6543 | 0.0059 | 0.0000 |
| stage03 | random noise | 2/255 | 0.6436 | 0.6475 | 0.2308 |

## 四、结果分析

1. `epsilon=0` 时 clean accuracy 与 adversarial accuracy 一致，说明评测脚本基本可信。
2. random noise `2/255` 下 accuracy 为 0.6475，几乎不影响模型表现。
3. FGSM 即使只有 `0.5/255`，accuracy 也下降到 0.0342，说明模型对梯度方向扰动非常敏感。
4. PGD-3 `1/255` 下 accuracy 进一步下降到 0.0059，说明迭代攻击更强。
5. 多个对抗实验的 worst-class accuracy 均为 0，说明医疗图像 baseline 的最差类别鲁棒性仍然很弱。

## 五、本周结论

本周完成了 ISIC 2019 clean baseline 的轻量对抗评测。结果表明，模型对普通随机扰动较稳定，但对 FGSM / PGD 这类基于梯度方向的对抗扰动非常脆弱。该结果可以作为后续 FAIR-TAT-like 对抗训练迁移到 ISIC 2019 的基准对照。

## 六、下周计划

1. 在 ISIC 2019 上尝试小规模 FGSM adversarial training。
2. 对比 clean baseline 与 adversarial training 后的 FGSM / PGD 鲁棒性。
3. 继续关注 worst-class robust accuracy 和 anatom-site worst-group accuracy。
4. 若时间允许，再尝试 targeted PGD 版本，为 FAIR-TAT-like 医疗图像迁移做准备。

## 七、材料位置

实验材料已整理到项目目录中：

- 对抗评测脚本：`scripts/evaluate_isic2019_adversarial.py`
- 第八周计划：`week8/docs/week8_task_plan.md`
- 主要实验：`week8/experiments/stage01_isic2019_fgsm_eval_eps2_255` 至 `week8/experiments/stage06_isic2019_fgsm_eval_eps0_sanity`
- 本周周报：`week8/docs/weekly_report.md`
