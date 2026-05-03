# FAIR-TAT-like CIFAR-10 阶段实验结果

## 实验设置

### stage02：5k 小规模实验

- 数据集：CIFAR-10
- 训练子集：5000
- 测试子集：1000
- 对抗评测子集：512
- 训练轮数：3
- warmup：1 epoch clean training
- 攻击：targeted PGD
- `epsilon=8/255`
- `alpha=2/255`
- `pgd_steps=3`
- `adv_weight=0.5`
- `prior_smoothing=5.0`
- 设备：CPU

### stage03：20k 扩大规模实验

- 数据集：CIFAR-10
- 训练子集：20000
- 测试子集：2000
- 对抗评测子集：1024
- 训练轮数：5
- warmup：1 epoch clean training
- 攻击：targeted PGD
- `epsilon=8/255`
- `alpha=2/255`
- `pgd_steps=5`
- `adv_weight=0.5`
- `prior_smoothing=5.0`
- 设备：CPU

### stage04：完整 CIFAR-10 实验

- 数据集：CIFAR-10
- 训练集：完整训练集
- 测试子集：5000
- 对抗评测子集：2048
- 训练轮数：6
- warmup：1 epoch clean training
- 攻击：targeted PGD
- `epsilon=8/255`
- `alpha=2/255`
- `pgd_steps=5`
- `adv_weight=0.5`
- `prior_smoothing=5.0`
- 设备：CPU

## 结果表

| 方法 | clean acc | clean worst-class acc | FGSM robust acc | FGSM worst-class robust acc | targeted PGD original acc | targeted PGD worst-class robust acc | target success rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| week1 clean baseline 5k | 0.4480 | 0.2727 | 0.0781 | 未记录 | 0.2539 | 未记录 | 0.3496 |
| week1 fixed-target PGD AT 小实验 | 0.3580 | 0.1798 | 0.0898 | 未记录 | 0.3691 | 未记录 | 0.2852 |
| week2 FAIR-TAT-like 5k | 0.4660 | 0.2584 | 0.1113 | 0.0000 | 0.3613 | 0.1765 | 0.3047 |
| week2 FAIR-TAT-like 20k | 0.5830 | 0.3175 | 0.1748 | 0.0388 | 0.4082 | 0.2000 | 0.3594 |
| week2 FAIR-TAT-like full | 0.6670 | 0.4843 | 0.1919 | 0.0650 | 0.4561 | 0.2621 | 0.3481 |
| stage05a adv_weight=0.3 20k | 0.5995 | 0.3333 | 0.1455 | 0.0485 | 0.3740 | 0.1889 | 0.4199 |
| stage05b adv_weight=0.7 20k | 0.5690 | 0.3016 | 0.1943 | 0.0485 | 0.4209 | 0.2222 | 0.3340 |
| stage06a smoothing=1.0 20k | 0.5890 | 0.3280 | 0.1738 | 0.0556 | 0.4072 | 0.2000 | 0.3633 |
| stage06b smoothing=10.0 20k | 0.5880 | 0.3333 | 0.1836 | 0.0485 | 0.4170 | 0.2182 | 0.3584 |
| stage07b PGD-7 full | 0.6656 | 0.4686 | 0.1919 | 0.0650 | 0.4624 | 0.2573 | 0.3555 |
| stage08a adv0.7 smooth10 full | 0.6558 | 0.4686 | 0.2090 | 0.0806 | 0.4761 | **0.2961** | 0.3184 |
| stage08b adv0.7 smooth10 PGD-7 full | 0.6494 | 0.4529 | 0.2153 | **0.0885** | **0.4844** | 0.2767 | 0.3159 |

## 分析

相比第一周 clean baseline，FAIR-TAT-like 5k 在 targeted PGD original accuracy 上明显更高，说明 targeted adversarial training 对相同类型攻击具有防御效果。同时 clean accuracy 也没有下降，反而达到 `0.4660`。

相比第一周固定目标 targeted PGD AT，FAIR-TAT-like 的目标类别不再固定，而是由 false positive prior 采样。当前 targeted PGD original accuracy 略低于固定目标 PGD AT 小实验，但 clean accuracy 和 clean worst-class accuracy 更高，说明动态 target prior 版本具备继续扩大训练的价值。

20k 扩大规模实验进一步提升了整体表现：clean accuracy 从 `0.4660` 提升到 `0.5830`，targeted PGD original accuracy 从 `0.3613` 提升到 `0.4082`，targeted PGD worst-class robust accuracy 从 `0.1765` 提升到 `0.2000`。这说明当前 FAIR-TAT-like 入口在更大训练数据上具备继续推进的价值。

完整 CIFAR-10 实验继续提升了结果：clean accuracy 达到 `0.6670`，clean worst-class accuracy 达到 `0.4843`，targeted PGD original accuracy 达到 `0.4561`，targeted PGD worst-class robust accuracy 达到 `0.2621`。这组结果可以作为第二周 FAIR-TAT-like 复现的主结果。

目前最主要的问题是 FGSM worst-class robust accuracy 仍然偏低。虽然 full 实验已经从 5k 的 `0.0000` 提升到 `0.0650`，但 targeted PGD 训练不能直接保证所有攻击类型下的 worst-class 表现。后续需要继续比较 PGD steps、`adv_weight` 和 `prior_smoothing`，并考虑加入 untargeted attack 的混合训练或更强 backbone。

stage05 比较了 `adv_weight=0.3` 和 `adv_weight=0.7`。`adv_weight=0.3` 的 clean accuracy 较高，但 targeted PGD 鲁棒性下降；`adv_weight=0.7` 的 targeted PGD original accuracy 和 targeted PGD worst-class robust accuracy 均优于 20k 默认设置，说明提高 adversarial loss 权重确实有助于 targeted PGD 威胁模型下的鲁棒性。不过 stage05b 仍未超过 full 基线的 `0.2621`，所以暂不作为 full 复训候选。

stage06 比较了 `prior_smoothing=1.0` 和 `prior_smoothing=10.0`。`smoothing=10.0` 的 targeted PGD worst-class robust accuracy 达到 `0.2182`，略高于 20k 默认设置，说明更平滑的 target prior 有助于提升 targeted PGD 最差类鲁棒性。

stage07 直接使用 full 数据测试 PGD-7。PGD-7 提高了 targeted PGD original accuracy，但 targeted PGD worst-class robust accuracy 为 `0.2573`，略低于 stage04 full 的 `0.2621`，因此单独提高 PGD 步数不是当前最优方向。

stage08 将 `adv_weight=0.7` 与 `prior_smoothing=10.0` 组合后进行 full 训练，得到了当前最好的结果。stage08a 的 targeted PGD worst-class robust accuracy 达到 `0.2961`，相比 stage04 full 的 `0.2621` 有明显提升；stage08b 的 FGSM worst-class robust accuracy 达到 `0.0885`，targeted PGD original accuracy 达到 `0.4844`，但 targeted PGD worst-class robust accuracy 低于 stage08a。因此当前建议把 stage08a 作为第二周最佳 FAIR-TAT-like 模型。
