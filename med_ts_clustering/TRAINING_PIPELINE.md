# 三阶段训练 Pipeline v2（模块化版本）

本版本将原来的 `trainer.py` 拆分为 4 个模块：

```
config.yaml           # 所有参数统一存放
pretrain_trainer.py   # MRM 预训练
DEC_trainer.py        # DEC 微调
inference.py          # 推理与聚类导出
pipeline.py           # 三阶段训练主入口（保留统一接口）
```

## 一键运行三阶段训练

```python
from pipeline import train_full_pipeline
from dataset.data import FeatureConfig
import yaml

cfg = yaml.safe_load(open("config.yaml"))

feature_cfg = FeatureConfig(
    patient_id_col="stay_id",
    time_col="hour",
    cont_cols=[...],
    cat_cols=[...],
    static_cont_cols=[...],
    static_cat_cols=[...],
)

train_full_pipeline(cfg, feature_cfg)
```

## pipeline.py（自动调用三个阶段）

pipeline 执行顺序：

1. 加载配置与数据
2. 构建模型
3. 调用 `train_mrm_pretrain`
4. 调用 `train_dec`
5. 调用 `export_clusters`


# 三阶段训练 Pipeline 使用指南

## 概述

本系统实现了完整的三阶段训练 pipeline，用于时序数据的无监督聚类：

1. **阶段 1：MRM 自监督预训练** - 通过 Masked Row Modeling 学习时序模式
2. **阶段 2：KMeans 初始化** - 使用预训练 embedding 初始化 DEC 聚类中心
3. **阶段 3：DEC 聚类微调** - 通过 KL 散度优化聚类边界

## 快速开始

### 基本使用

```python
from med_ts_clustering.dataset.data import FeatureConfig
from med_ts_clustering.trainer import train_mrm_dec_pipeline, export_cluster_assignments

# 配置特征
feature_cfg = FeatureConfig(
    patient_id_col="stay_id",
    time_col="hour",
    cont_cols=["heart_rate", "systolic_bp", "temperature"],  # 连续特征
    cat_cols=["gender", "diagnosis"],  # 分类特征
    static_cont_cols=["age", "weight"],  # 静态连续特征
    static_cat_cols=["admission_type"],  # 静态分类特征
)

# 运行三阶段训练
model, dataset = train_mrm_dec_pipeline(
    static_csv_path="path/to/static.csv",
    events_csv_path="path/to/events.csv",
    feature_cfg=feature_cfg,
    n_clusters=10,  # 聚类数量
    d_model=128,
    max_seq_len=128,
    batch_size=32,
    # Stage 1: MRM 预训练
    mrm_n_epochs=50,
    mrm_lr=1e-4,
    mrm_mask_ratio=0.2,  # 遮挡 20% 的字段
    # Stage 3: DEC 微调
    dec_n_epochs=50,
    dec_lr=1e-4,
    output_dir="outputs",
    save_every=10,
)

# 导出聚类结果
export_cluster_assignments(
    model=model,
    dataset=dataset,
    feature_cfg=feature_cfg,
    max_seq_len=128,
    batch_size=32,
    output_csv="cluster_assignments.csv",
)
```

## 详细参数说明

### `train_mrm_dec_pipeline` 参数

#### 必需参数
- `static_csv_path`: 静态特征 CSV 文件路径
- `events_csv_path`: 时序事件 CSV 文件路径
- `feature_cfg`: 特征配置对象
- `n_clusters`: 聚类数量

#### 模型参数
- `d_model`: 模型 embedding 维度（默认：128）
- `max_seq_len`: 最大序列长度（默认：128）
- `batch_size`: 批次大小（默认：32）
- `ft_kwargs`: FT-Transformer 额外参数（可选）
- `time_transformer_cfg`: Time Transformer 配置（可选）

#### 阶段 1：MRM 预训练参数
- `mrm_n_epochs`: MRM 预训练轮数（默认：50）
- `mrm_lr`: MRM 预训练学习率（默认：1e-4）
- `mrm_mask_ratio`: MRM 遮挡比例，范围 0.15-0.30（默认：0.2）

#### 阶段 3：DEC 微调参数
- `dec_n_epochs`: DEC 微调轮数（默认：50）
- `dec_lr`: DEC 微调学习率（默认：1e-4）

#### 其他参数
- `device`: 设备（"cuda" 或 "cpu"，默认自动检测）
- `output_dir`: 输出目录（默认："outputs_mrm_dec"）
- `save_every`: 每 N 轮保存一次 checkpoint（默认：10）

## 输出文件

训练完成后，在 `output_dir` 目录下会生成：

1. `mrm_pretrained.pt` - MRM 预训练后的模型
2. `dec_finetune_epoch_*.pt` - DEC 微调过程中的 checkpoints
3. `final_model.pt` - 最终训练完成的模型

## 数据格式要求

### 静态数据表（static.csv）
必须包含 `patient_id_col` 指定的患者 ID 列，以及所有 `static_cont_cols` 和 `static_cat_cols` 指定的列。

### 时序事件表（events.csv）
必须包含：
- `patient_id_col` 指定的患者 ID 列
- `time_col` 指定的时间列
- 所有 `cont_cols` 和 `cat_cols` 指定的列

### 缺失值处理
- 连续特征：缺失值会用该列的 median 填充（如果全为 NaN 则用 0）
- 分类特征：缺失值会被编码为特殊 token "___NA___"
- 系统会自动记录原始缺失位置，用于 MRM 训练

## 训练阶段详解

### 阶段 1：MRM 自监督预训练

**目标**：让模型学习时序数据中的模式、多变量相关性和缺失模式。

**过程**：
1. 对每一行随机遮挡 15-30% 的字段（由 `mrm_mask_ratio` 控制）
2. 保留真实缺失 mask（`missing_mask`）和 MRM 遮挡 mask（`mrm_mask`）
3. 模型尝试重建被遮挡的字段
4. 连续值使用 MSE loss，分类值使用 CrossEntropy loss
5. 只对 MRM 遮挡（非原始缺失）的字段计算 loss

**输出**：稳定的 encoder，能生成高质量的行表示 `h_t`

### 阶段 2：KMeans 初始化

**目标**：为 DEC 提供合理的初始聚类中心。

**过程**：
1. 使用预训练后的 encoder 对所有样本生成 embedding
2. 对所有 embedding 运行 KMeans 聚类
3. 将 KMeans 的中心点写入 DEC 的 `cluster_centers`

**输出**：初始化好的 DEC 聚类中心

### 阶段 3：DEC 聚类微调

**目标**：通过 KL 散度优化聚类边界，使簇更清晰。

**过程**：
1. 冻结 ReconstructionHead（不再使用）
2. Forward 得到软分配 `q_ij`
3. 构造目标分布 `p_ij`（增强 confident 分配）
4. 计算 DEC loss：`L = KL(P || Q)`
5. 反向传播更新 encoder 和 DEC centers
6. Encoder 使用较小学习率（`dec_lr * 0.1`），DEC centers 使用正常学习率

**输出**：每一行的最终 `cluster_id` 和优化后的 embedding 空间

## 高级用法

### 自定义 FT-Transformer 配置

```python
ft_kwargs = {
    "d_block": 192,
    "n_blocks": 4,
    "attention_n_heads": 8,
    "attention_dropout": 0.1,
    "ffn_dropout": 0.1,
}

model, dataset = train_mrm_dec_pipeline(
    ...,
    ft_kwargs=ft_kwargs,
)
```

### 自定义 Time Transformer 配置

```python
time_transformer_cfg = {
    "n_heads": 4,
    "n_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "use_time_scalar": True,
}

model, dataset = train_mrm_dec_pipeline(
    ...,
    time_transformer_cfg=time_transformer_cfg,
)
```

### 仅运行特定阶段

如果需要分阶段训练，可以单独调用：

```python
from med_ts_clustering.trainer import train_mrm_pretrain, initialize_kmeans

# 只运行 MRM 预训练
train_mrm_pretrain(model, dataloader, device, n_epochs=50)

# 只运行 KMeans 初始化
initialize_kmeans(model, dataloader, device)
```

## 输出结果格式

`cluster_assignments.csv` 包含以下列：
- `stay_id`（或你指定的 `patient_id_col`）
- `hour`（或你指定的 `time_col`）
- `cluster_id`：该行的聚类 ID（0 到 n_clusters-1）

同时会生成 `cluster_assignments_embeddings.npy`，包含所有行的 embedding 向量，可用于可视化。

## 注意事项

1. **内存使用**：如果数据量很大，可能需要调整 `batch_size` 或 `max_seq_len`
2. **训练时间**：三阶段训练需要较长时间，建议使用 GPU
3. **聚类数量**：`n_clusters` 的选择需要根据数据特点调整，可以使用肘部法则等方法
4. **缺失值**：系统会自动处理缺失值，但建议在数据预处理阶段尽量保证数据质量

## 故障排除

### 问题：内存不足
- 减小 `batch_size`
- 减小 `max_seq_len`
- 减小 `d_model`

### 问题：训练不稳定
- 减小学习率
- 增加 `mrm_n_epochs` 确保预训练充分
- 检查数据质量

### 问题：聚类效果不佳
- 调整 `n_clusters`
- 增加 `dec_n_epochs`
- 尝试不同的 `mrm_mask_ratio`（0.15-0.30）

