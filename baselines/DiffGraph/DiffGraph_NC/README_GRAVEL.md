# DiffGraph for GRAVEL Datasets

本文档说明如何在 GRAVEL 数据集（pdns, minta, iochg）上训练 DiffGraph 模型进行**二分类**任务（benign vs malicious）。

## 修改内容

### 1. 数据加载适配（`DataHandler.py`）

- **添加了 `load_gravel_data()` 方法**：从 `baseline_trainer.py` 的 `get_g()` 函数加载数据
- **二分类标签处理**：
  - 原始标签：0=benign, 1=malicious, 2=n_a (unlabeled)
  - 转换为 one-hot 二分类标签：`[1,0]` 表示 benign，`[0,1]` 表示 malicious
  - 自动过滤掉未标记的节点（label==2）
  
- **元路径提取**：
  - **pdns**: 3个元路径
    1. domain-ip-domain (通过 IP 连接的域名)
    2. domain-similar-domain (相似域名)
    3. domain-apex-domain (顶级域名关系)
  
  - **minta**: 3个元路径
    1. domain-ip-domain
    2. domain-similar-domain  
    3. domain-apex-domain
  
  - **iochg**: 3个元路径
    1. domain-ip-domain
    2. domain-domain (类型7，直接关系)
    3. domain-domain (类型9，间接关系)

### 2. AUC 计算修复（`main.py`）

- **修复了二分类 AUC 计算错误**
- 添加了错误处理，当测试集类别数与预测类别数不匹配时自动调整
- 改进了日志输出格式

### 3. 参数配置（`params.py`）

- 更新了 `--data` 参数说明，支持：`aminer, DBLP, Freebase, pdns, minta, iochg`

## 为什么需要元路径？

**DiffGraph 是基于元路径的异构图扩散模型**。元路径的作用是：

1. **捕获不同语义关系**：每个元路径代表一种不同类型的连接方式
   - domain-ip-domain：两个域名解析到相同IP
   - domain-similar-domain：域名字符串相似
   - domain-apex-domain：共享顶级域名

2. **多视角学习**：DiffGraph 从不同元路径视角学习节点表示，然后融合这些表示

3. **降维处理**：将复杂的异构图转换为多个简单的同构邻接矩阵

## 使用方法

### 测试数据加载

```bash
# 测试二分类数据加载是否正常
python test_binary_classification.py
```

**预期输出**：
```
✓ Data loaded successfully!
  - Feature shape: torch.Size([897635, 16])
  - Labels shape: torch.Size([897635, 2]) (should be [num_nodes, 2] for binary)
  - Number of metapaths: 3
  - Benign: 29140, Malicious: 59813, Unlabeled: 808682
```

### 训练模型

```bash
# PDNS 数据集
python main.py --data pdns --lr 3e-3 --batch 256 --epoch 100 --steps 200 --noise_scale 1e-5

# MINTA 数据集
python main.py --data minta --lr 3e-3 --batch 256 --epoch 100 --steps 200 --noise_scale 1e-5

# IOCHG 数据集
python main.py --data iochg --lr 3e-3 --batch 256 --epoch 100 --steps 200 --noise_scale 1e-5
```

### 重要参数说明

- `--data`: 数据集名称（pdns/minta/iochg）
- `--lr`: 学习率（默认 3e-3）
- `--batch`: 批大小（建议 128-512）
- `--epoch`: 训练轮数（建议 100-200）
- `--steps`: 扩散步数（建议 50-200）
- `--noise_scale`: 噪声缩放（建议 1e-5 到 1e-4）
- `--ratio`: 训练数据比例（默认 [20, 40, 60]）

### 训练输出

训练日志保存在 `History/` 目录，模型保存在 `Models/` 目录。

最终输出示例：
```
[Classification] Macro-F1: 0.8567 var: 0.0123  
                 Micro-F1_mean: 0.8734 var: 0.0098  
                 auc: 0.9012 var: 0.0087
```

## 数据统计

### PDNS
- 总节点数：897,635 个域名
- 标记数据：88,953（benign: 29,140, malicious: 59,813）
- 训练/验证/测试：87,454 / 230 / 230
- 元路径数量：3

### MINTA  
- 总节点数：377,554 个域名
- 标记数据：约 28,500
- 训练/验证/测试：28,148 / 180 / 180
- 元路径数量：3

### IOCHG
- 总节点数：2,620,775 个域名
- 标记数据：约 791,000
- 训练/验证/测试：774,843 / 8,028 / 8,028
- 元路径数量：3

## 故障排除

### 错误：`Number of classes in y_true not equal to the number of columns in 'y_score'`

**解决方案**：这个错误已经在新版本中修复。请确保：
1. 使用最新版本的 `main.py` 和 `DataHandler.py`
2. 检查标签形状是否为 `[num_nodes, 2]`（二分类）
3. 重新运行训练命令

### 内存不足

如果遇到 OOM 错误：
- 减小 `--batch` 大小（如 128 或 64）
- 减小 `--steps` 扩散步数（如 50）

### GPU 使用

默认使用 `cuda:0`。如需更改：
- 修改 `DataHandler.py` 中的 `device` 变量
- 或使用 `CUDA_VISIBLE_DEVICES` 环境变量

## 技术细节

### 元路径邻接矩阵计算

对于多跳元路径（如 domain-ip-domain）：
1. 构建第一跳矩阵：domain → ip
2. 构建第二跳矩阵：ip → domain  
3. 矩阵相乘得到 domain → domain 的连接
4. 转换为二值邻接矩阵（有连接=1，无连接=0）
5. 添加自环（self-loop）

### 标签过滤

只使用有明确标签的节点进行训练：
```python
labeled_mask = (labels_np == 0) | (labels_np == 1)  # 只保留 benign 和 malicious
train_idx_filtered = filter_indices(train_idx, labeled_mask)
```

## 相关文件

- `DataHandler.py`: 数据加载和元路径提取
- `main.py`: 训练主程序
- `params.py`: 参数配置
- `Model.py`: DiffGraph 模型定义
- `baseline_trainer.py`: GRAVEL 数据生成函数 `get_g()`

## 引用

如果使用此代码，请引用：
- DiffGraph 原始论文
- GRAVEL 项目

---

**最后更新**: 2025-10-08
**状态**: ✅ 已测试并可用于二分类任务
