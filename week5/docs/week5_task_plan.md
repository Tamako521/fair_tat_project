# 第五周任务计划

姓名：郭家棋

## 一、本周目标

前四周主要在 CIFAR-10 上完成 FAIR-TAT-like 复现、per-class 鲁棒性调节和 ResNet18 full 实验。本周开始扩展训练数据集，验证当前方法是否能迁移到类别更多、任务更难的数据集。

本周第一目标：

**接入 CIFAR-100，并完成最小测试与 AI Studio stage01 训练任务准备。**

## 二、任务安排

1. 扩展数据加载模块，支持 `cifar10` 与 `cifar100`。
2. 修改训练入口，增加 `--dataset` 参数。
3. 在本地下载 CIFAR-100，跑一个最小训练测试，确认代码流程可用。
4. 整理需要同步到 AI Studio 的代码文件。
5. 设计 `stage01_cifar100_resnet18_smoke`，在云端 GPU 上跑较小规模验证。
6. 若 stage01 成功，再扩大到 CIFAR-100 20k 或 full。

## 三、实验顺序

| 阶段 | 实验名称 | 目的 |
|---|---|---|
| local smoke | `week5_local_cifar100_smoke` | 本地最小数据测试，确认 CIFAR-100 接入成功 |
| stage01 | `stage01_cifar100_resnet18_smoke` | AI Studio 上验证 CIFAR-100 + ResNet18 + per-class 调节 |
| stage02 | `stage02_cifar100_resnet18_20k` | 扩大训练规模 |
| stage03 | `stage03_cifar100_resnet18_full` | 使用完整 CIFAR-100 训练集 |

## 四、本周重点指标

CIFAR-100 有 100 个类别，比 CIFAR-10 更难，因此重点不只看 clean accuracy，也要看：

- clean accuracy
- clean worst-class accuracy
- targeted PGD original accuracy
- targeted PGD worst-class accuracy
- untargeted PGD robust accuracy
- untargeted PGD worst-class accuracy

## 五、预期产出

1. 支持 CIFAR-100 的训练入口。
2. 一个本地 CIFAR-100 最小测试结果。
3. 一个 AI Studio stage01 训练命令。
4. 一份需要同步到 AI Studio 的文件清单。
5. 为后续 STL-10 和 ISIC 2019 迁移做准备。
