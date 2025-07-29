# GRAVEL: Graph Representation Learning with Visual Enhancement

## 项目简介

GRAVEL是一个基于图神经网络的图表示学习项目，结合了视觉增强技术来提升图数据的表示能力。

## 项目结构

```
GRAVEL/
├── GRAVEL_GraphEncoder.py      # 图编码器实现
├── GRAVEL_endtoend.py          # 端到端训练脚本
├── gravel_backbone.py          # 骨干网络实现
├── contrastive_learning.py     # 对比学习实现
├── fintune_GRAVEL.py           # GRAVEL微调脚本
├── finetune_baselines.py       # 基线模型微调脚本
├── evaluate_finetune_gravel.py # 评估脚本
├── split_dataset.py            # 数据集分割脚本
├── baselines/                  # 基线模型实现
├── dataset/                    # 数据集目录
├── checkpoint/                 # 模型检查点
├── feats/                      # 特征文件
└── plot/                       # 可视化图表
```

## 主要功能

- **图编码器**: 实现基于GNN的图表示学习
- **对比学习**: 支持多种对比学习策略
- **端到端训练**: 完整的训练流程
- **模型微调**: 支持预训练模型的微调
- **基线对比**: 与多种基线模型进行对比

## 环境要求

- Python 3.7+
- PyTorch
- PyTorch Geometric
- 其他依赖见requirements.txt

## 使用方法

1. 安装依赖
2. 准备数据集
3. 运行训练脚本
4. 评估模型性能

## 许可证

[在此添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。 